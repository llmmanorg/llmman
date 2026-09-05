//! The dynamically-loaded CUDA/HIP/Vulkan probes behind
//! [`super::detect_gpu_api_uncontained`]. Its own file purely so the
//! whole section hangs off the one `#[cfg]` on its `mod` declaration:
//! macOS has none of these three runtimes (it uses Metal — see
//! [`super::detect_macos`]), so gating them one attribute at a time is
//! what let ~22 `dead_code` warnings accumulate there unnoticed.
//!
//! Mirrors `cuda/probe.c`, `rocm/probe.cc` and `vulkan/probe.c` from
//! ggml-org/llama-install.sh — see `hostgpu.rs` for why these are
//! reached by runtime dynamic loading rather than as probe binaries.

use std::ffi::c_void;
use std::os::raw::c_char;

use libloading::{Library, Symbol};

use super::{igpu_enabled, HostGpu};

// ---------------------------------------------------------------------------
// Dynamic library loading — one candidate list per backend/OS, mirroring
// which shared library cuda/probe.c, rocm/probe.cc, and vulkan/probe.c
// each link against.
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn open_cuda_lib() -> Option<Library> {
    unsafe { Library::new("nvcuda.dll").ok() }
}

#[cfg(target_os = "linux")]
fn open_cuda_lib() -> Option<Library> {
    unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .ok()
    }
}

#[cfg(target_os = "windows")]
fn open_hip_lib() -> Option<Library> {
    unsafe {
        Library::new("amdhip64_6.dll")
            .or_else(|_| Library::new("amdhip64.dll"))
            .ok()
    }
}

#[cfg(target_os = "linux")]
fn open_hip_lib() -> Option<Library> {
    unsafe {
        Library::new("libamdhip64.so.6")
            .or_else(|_| Library::new("libamdhip64.so.5"))
            .or_else(|_| Library::new("libamdhip64.so"))
            .ok()
    }
}

#[cfg(target_os = "windows")]
fn open_vulkan_lib() -> Option<Library> {
    unsafe { Library::new("vulkan-1.dll").ok() }
}

#[cfg(target_os = "linux")]
fn open_vulkan_lib() -> Option<Library> {
    unsafe {
        Library::new("libvulkan.so.1")
            .or_else(|_| Library::new("libvulkan.so"))
            .ok()
    }
}

// ---------------------------------------------------------------------------
// CUDA — mirrors cuda/probe.c: cuDriverGetVersion, cuInit,
// cuDeviceGetCount, then (best-effort) cuDeviceGet/cuDeviceGetAttribute
// per device. Unlike probe.c, there's no PROBE_ARCH/PROBE_VERSION
// best-arch match against llama-app's own per-SM-arch build matrix —
// llama.cpp's official GitHub releases only split CUDA builds by major
// version (12 vs. 13, from the driver's own reported version), not by SM
// architecture — so the per-device compute-capability query below is
// kept only to confirm those same calls succeed against a real device,
// not to select anything.
// ---------------------------------------------------------------------------

const CUDA_SUCCESS: i32 = 0;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

/// CUDA's driver version int encodes `major*1000 + minor*10` (e.g. 12040
/// = "12.4", 13030 = "13.3") — this is `cuDriverGetVersion`'s own
/// documented format, the same value `nvidia-smi`'s human-readable "CUDA
/// Version: X.Y" header is derived from.
fn cuda_major_from_driver_version(driver_version: i32) -> u32 {
    (driver_version / 1000).max(0) as u32
}

/// `Some((kind, total_vram_bytes))` if at least one CUDA device is
/// present. Sums every visible device's memory (via
/// `cuDeviceTotalMem_v2`), not just device 0's, so a multi-GPU host's
/// combined VRAM weighs it correctly in aggregation (see
/// `hostgpu::memory_bytes`) — matching `detect_vulkan_inner`'s existing
/// multi-device summation below.
pub(super) fn detect_cuda() -> Option<(HostGpu, u64)> {
    let lib = open_cuda_lib()?;
    unsafe {
        let cu_driver_get_version: Symbol<unsafe extern "C" fn(*mut i32) -> i32> =
            lib.get(b"cuDriverGetVersion\0").ok()?;
        let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> = lib.get(b"cuInit\0").ok()?;
        let cu_device_get_count: Symbol<unsafe extern "C" fn(*mut i32) -> i32> =
            lib.get(b"cuDeviceGetCount\0").ok()?;

        let mut driver_version: i32 = 0;
        if cu_driver_get_version(&mut driver_version) != CUDA_SUCCESS {
            return None;
        }
        if cu_init(0) != CUDA_SUCCESS {
            return None;
        }
        let mut count: i32 = 0;
        if cu_device_get_count(&mut count) != CUDA_SUCCESS || count == 0 {
            return None;
        }

        let mut vram: u64 = 0;
        if let (Ok(cu_device_get), Ok(cu_device_get_attribute)) = (
            lib.get::<unsafe extern "C" fn(*mut i32, i32) -> i32>(b"cuDeviceGet\0"),
            lib.get::<unsafe extern "C" fn(*mut i32, i32, i32) -> i32>(b"cuDeviceGetAttribute\0"),
        ) {
            let cu_device_total_mem = lib
                .get::<unsafe extern "C" fn(*mut u64, i32) -> i32>(b"cuDeviceTotalMem_v2\0")
                .ok();
            for index in 0..count {
                let mut device: i32 = 0;
                if cu_device_get(&mut device, index) != CUDA_SUCCESS {
                    continue;
                }
                let mut major: i32 = 0;
                let mut minor: i32 = 0;
                cu_device_get_attribute(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                );
                cu_device_get_attribute(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                );
                crate::debug_log!("CUDA device {index} compute capability: {major}.{minor}");

                if let Some(cu_device_total_mem) = &cu_device_total_mem {
                    let mut bytes: u64 = 0;
                    if cu_device_total_mem(&mut bytes, device) == CUDA_SUCCESS {
                        vram += bytes;
                    }
                }
            }
        }

        Some((
            HostGpu::Cuda {
                major: cuda_major_from_driver_version(driver_version),
            },
            vram,
        ))
    }
}

// ---------------------------------------------------------------------------
// ROCm — mirrors rocm/probe.cc: hipGetDeviceCount. probe.cc goes on to
// read hipGetDeviceProperties(...).gcnArchName; skipped here since
// llmman's Rocm variant carries no arch detail (llama.cpp's ROCm release
// asset isn't split by GPU arch either) and hipDeviceProp_t's layout
// isn't ABI-frozen the way CUDA/Vulkan's structs used elsewhere in this
// file are (it has grown fields across ROCm releases), making it unsafe
// to lay out by hand here.
// ---------------------------------------------------------------------------

const HIP_SUCCESS: i32 = 0;

/// `Some(total_vram_bytes)` if at least one HIP device is present, `None`
/// if not. Sums every device's memory rather than just device 0's — see
/// `detect_cuda`'s doc comment.
pub(super) fn detect_rocm() -> Option<u64> {
    let lib = open_hip_lib()?;
    unsafe {
        let hip_get_device_count = lib
            .get::<unsafe extern "C" fn(*mut i32) -> i32>(b"hipGetDeviceCount\0")
            .ok()?;
        let mut count: i32 = 0;
        if hip_get_device_count(&mut count) != HIP_SUCCESS || count == 0 {
            return None;
        }

        let mut vram: u64 = 0;
        if let (Ok(hip_set_device), Ok(hip_mem_get_info)) = (
            lib.get::<unsafe extern "C" fn(i32) -> i32>(b"hipSetDevice\0"),
            lib.get::<unsafe extern "C" fn(*mut u64, *mut u64) -> i32>(b"hipMemGetInfo\0"),
        ) {
            for index in 0..count {
                if hip_set_device(index) != HIP_SUCCESS {
                    continue;
                }
                let (mut free, mut total): (u64, u64) = (0, 0);
                if hip_mem_get_info(&mut free, &mut total) == HIP_SUCCESS {
                    vram += total;
                }
            }
        }
        Some(vram)
    }
}

// ---------------------------------------------------------------------------
// Vulkan — mirrors vulkan/probe.c: vkCreateInstance,
// vkEnumeratePhysicalDevices, vkGetPhysicalDeviceProperties (skip
// VK_PHYSICAL_DEVICE_TYPE_CPU always, and _INTEGRATED_GPU unless
// LLMMAN_IGPU_ENABLE is set — llmman's own addition, not in probe.c —
// and require apiVersion >= 1.2),
// vkGetPhysicalDeviceQueueFamilyProperties (require a queue family with
// both COMPUTE and TRANSFER bits set).
// ---------------------------------------------------------------------------

const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: i32 = 1;
const VK_SUCCESS: i32 = 0;
const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: i32 = 1;
const VK_PHYSICAL_DEVICE_TYPE_CPU: i32 = 4;
const VK_QUEUE_COMPUTE_BIT: u32 = 0x2;
const VK_QUEUE_TRANSFER_BIT: u32 = 0x4;

fn vk_make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

type VkInstance = *mut c_void;
type VkPhysicalDevice = *mut c_void;

#[repr(C)]
struct VkInstanceCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    p_application_info: *const c_void,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VkExtent3D {
    width: u32,
    height: u32,
    depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VkQueueFamilyProperties {
    queue_flags: u32,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: VkExtent3D,
}

/// A generously oversized, 8-byte-aligned stand-in for
/// `VkPhysicalDeviceProperties` (~824 bytes on a 64-bit build, dominated
/// by the large `VkPhysicalDeviceLimits` sub-struct). The Vulkan spec
/// freezes this struct's layout for ABI compatibility, so it's safe for
/// `vkGetPhysicalDeviceProperties` to write its real, full contents into
/// any buffer at least this big and this aligned — this only ever reads
/// back the two leading fields it actually needs (`apiVersion` at offset
/// 0, `deviceType` at offset 16), exactly what `vulkan/probe.c` reads off
/// its own fully-typed `VkPhysicalDeviceProperties`.
#[repr(C, align(8))]
struct RawPhysicalDeviceProperties([u8; 1024]);

const VK_MAX_MEMORY_HEAPS: usize = 16;
const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: u32 = 0x1;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VkMemoryHeap {
    size: u64,
    flags: u32,
    _pad: u32,
}

/// `VkPhysicalDeviceMemoryProperties`, minus its 32-entry `memoryTypes`
/// array (unused here) — safe to overwrite as a stand-in the same way
/// [`RawPhysicalDeviceProperties`] is: `vkGetPhysicalDeviceMemoryProperties`
/// only ever reads/writes the struct's fixed, spec-frozen layout, and
/// this reads back just `memoryHeapCount`/`memoryHeaps`, past a
/// same-sized raw padding block standing in for `memoryTypes`.
#[repr(C)]
struct RawPhysicalDeviceMemoryProperties {
    memory_type_count: u32,
    _memory_types_padding: [u8; 32 * 8],
    memory_heap_count: u32,
    memory_heaps: [VkMemoryHeap; VK_MAX_MEMORY_HEAPS],
}

pub(super) fn detect_vulkan() -> Option<u64> {
    let lib = open_vulkan_lib()?;
    unsafe { detect_vulkan_inner(&lib).unwrap_or(None) }
}

unsafe fn detect_vulkan_inner(lib: &Library) -> Option<Option<u64>> {
    let vk_create_instance: Symbol<
        unsafe extern "C" fn(*const VkInstanceCreateInfo, *const c_void, *mut VkInstance) -> i32,
    > = lib.get(b"vkCreateInstance\0").ok()?;
    let vk_destroy_instance: Symbol<unsafe extern "C" fn(VkInstance, *const c_void)> =
        lib.get(b"vkDestroyInstance\0").ok()?;
    let vk_enumerate_physical_devices: Symbol<
        unsafe extern "C" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> i32,
    > = lib.get(b"vkEnumeratePhysicalDevices\0").ok()?;
    let vk_get_physical_device_properties: Symbol<unsafe extern "C" fn(VkPhysicalDevice, *mut u8)> =
        lib.get(b"vkGetPhysicalDeviceProperties\0").ok()?;
    let vk_get_physical_device_queue_family_properties: Symbol<
        unsafe extern "C" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties),
    > = lib
        .get(b"vkGetPhysicalDeviceQueueFamilyProperties\0")
        .ok()?;
    let vk_get_physical_device_memory_properties: Symbol<
        unsafe extern "C" fn(VkPhysicalDevice, *mut RawPhysicalDeviceMemoryProperties),
    > = lib.get(b"vkGetPhysicalDeviceMemoryProperties\0").ok()?;

    let create_info = VkInstanceCreateInfo {
        s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: 0,
        p_application_info: std::ptr::null(),
        enabled_layer_count: 0,
        pp_enabled_layer_names: std::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: std::ptr::null(),
    };
    let mut instance: VkInstance = std::ptr::null_mut();
    if vk_create_instance(&create_info, std::ptr::null(), &mut instance) != VK_SUCCESS {
        return Some(None);
    }

    let mut count: u32 = 0;
    let found = if vk_enumerate_physical_devices(instance, &mut count, std::ptr::null_mut())
        != VK_SUCCESS
        || count == 0
    {
        None
    } else {
        let mut devices: Vec<VkPhysicalDevice> = vec![std::ptr::null_mut(); count as usize];
        if vk_enumerate_physical_devices(instance, &mut count, devices.as_mut_ptr()) != VK_SUCCESS {
            None
        } else {
            let mut found: Option<u64> = None;
            for &device in &devices {
                let mut props_buf = RawPhysicalDeviceProperties([0u8; 1024]);
                vk_get_physical_device_properties(device, props_buf.0.as_mut_ptr());
                let api_version = u32::from_ne_bytes(props_buf.0[0..4].try_into().unwrap());
                let device_type = i32::from_ne_bytes(props_buf.0[16..20].try_into().unwrap());

                if device_type == VK_PHYSICAL_DEVICE_TYPE_CPU {
                    continue;
                }
                if device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && !igpu_enabled() {
                    continue;
                }
                if api_version < vk_make_api_version(0, 1, 2, 0) {
                    continue;
                }

                let mut qcount: u32 = 0;
                vk_get_physical_device_queue_family_properties(
                    device,
                    &mut qcount,
                    std::ptr::null_mut(),
                );
                if qcount == 0 {
                    continue;
                }
                let mut queues = vec![VkQueueFamilyProperties::default(); qcount as usize];
                vk_get_physical_device_queue_family_properties(
                    device,
                    &mut qcount,
                    queues.as_mut_ptr(),
                );

                let has_compute = queues.iter().any(|q| {
                    q.queue_flags & VK_QUEUE_COMPUTE_BIT != 0
                        && q.queue_flags & VK_QUEUE_TRANSFER_BIT != 0
                });
                if !has_compute {
                    continue;
                }

                let mut mem_props = RawPhysicalDeviceMemoryProperties {
                    memory_type_count: 0,
                    _memory_types_padding: [0u8; 32 * 8],
                    memory_heap_count: 0,
                    memory_heaps: [VkMemoryHeap::default(); VK_MAX_MEMORY_HEAPS],
                };
                vk_get_physical_device_memory_properties(device, &mut mem_props);
                let vram: u64 = mem_props.memory_heaps
                    [..(mem_props.memory_heap_count as usize).min(VK_MAX_MEMORY_HEAPS)]
                    .iter()
                    .filter(|h| h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0)
                    .map(|h| h.size)
                    .sum();
                found = Some(found.unwrap_or(0) + vram);
            }
            found
        }
    };

    vk_destroy_instance(instance, std::ptr::null());
    Some(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_major_from_driver_version_matches_documented_encoding() {
        assert_eq!(cuda_major_from_driver_version(12040), 12); // "12.4"
        assert_eq!(cuda_major_from_driver_version(13030), 13); // "13.3"
        assert_eq!(cuda_major_from_driver_version(0), 0);
    }

    #[test]
    fn vk_make_api_version_matches_the_vulkan_header_macro() {
        // VK_API_VERSION_1_2 per vulkan_core.h: VK_MAKE_API_VERSION(0, 1, 2, 0)
        assert_eq!(vk_make_api_version(0, 1, 2, 0), (1 << 22) | (2 << 12));
        assert_eq!(vk_make_api_version(0, 1, 0, 0), 1 << 22);
    }

    /// Exercises the real dynamic-loaded CUDA/HIP/Vulkan probes against
    /// whatever hardware/drivers are actually present on the machine
    /// running the test — not run by a plain `cargo test` since its
    /// result depends entirely on the host, but useful to eyeball
    /// directly. Run explicitly with `cargo test --bin llmman -- --ignored
    /// --nocapture detect_reports_this_hosts_real_hardware`.
    #[test]
    #[ignore = "result depends on this host's actual GPU/driver setup"]
    fn detect_reports_this_hosts_real_hardware() {
        // `detect()` itself is deliberately not exercised here:
        // `detect_gpu_api_isolated` re-execs `std::env::current_exe()`,
        // which under `cargo test` is this test harness binary, not the
        // real `llmman` CLI binary — its generated `main()` doesn't check
        // for PROBE_SUBPROCESS_ARG at all, so `detect()` only ever
        // reports HostGpu::None when run this way (a `cargo test`-only
        // artifact, not something a real caller of the compiled `llmman`
        // binary hits). To verify the real re-exec path end to end
        // instead, run this directly against a real build:
        //
        //   cargo build --bin llmman
        //   ./target/debug/llmman __hostgpu-probe
        //
        // — confirmed directly this way while writing this module.
        println!("detect_cuda() -> {:?}", detect_cuda());
        println!("detect_rocm() -> {:?}", detect_rocm());
        println!("detect_vulkan() -> {:?}", detect_vulkan());
        println!(
            "detect_with_vram() -> {:?}",
            super::super::detect_with_vram()
        );
    }
}
