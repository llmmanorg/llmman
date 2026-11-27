//! The manatee mark (build.rs downloads the PNGs; generate-assets.py makes them).

/// 32 px and 64 px: Windows takes the first, macOS the second, Linux offers both.
pub const TRAY: [&[u8]; 2] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/tray-32.png")),
    include_bytes!(concat!(env!("OUT_DIR"), "/tray-64.png")),
];

/// RGBA pixels, width and height. Panics on a bad PNG; a test decodes ours.
pub fn decode(png: &[u8]) -> (Vec<u8>, u32, u32) {
    let mut reader = png::Decoder::new(std::io::Cursor::new(png))
        .read_info()
        .expect("embedded PNG header");
    let mut rgba = vec![0; reader.output_buffer_size().expect("embedded PNG size")];
    let info = reader.next_frame(&mut rgba).expect("embedded PNG data");
    assert_eq!(
        (info.color_type, info.bit_depth),
        (png::ColorType::Rgba, png::BitDepth::Eight)
    );
    rgba.truncate(info.buffer_size());
    (rgba, info.width, info.height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_pngs_decode() {
        for (png, side) in TRAY.iter().zip([32, 64]) {
            let (rgba, w, h) = decode(png);
            assert_eq!((w, h), (side, side));
            assert_eq!(rgba.len(), (side * side * 4) as usize);
            assert_eq!(rgba[3], 0, "corners are transparent");
            assert!(rgba.chunks(4).any(|px| px[3] == 255), "the mark is opaque");
        }
    }
}
