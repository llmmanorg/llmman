//! macOS and Windows: a tray-icon on a tao event loop.

use std::sync::Arc;

use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{Icon, MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};

use crate::daemon::{Daemon, IDLE};
use crate::icon;

enum UserEvent {
    Tray(TrayIconEvent),
    Menu(MenuEvent),
    Tooltip(&'static str),
}

pub fn run() -> ! {
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
    let mut event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    #[cfg(target_os = "macos")]
    {
        use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
        event_loop.set_activation_policy(ActivationPolicy::Accessory);
    }

    let proxy = event_loop.create_proxy();
    TrayIconEvent::set_event_handler(Some({
        let proxy = proxy.clone();
        move |event| {
            let _ = proxy.send_event(UserEvent::Tray(event));
        }
    }));
    MenuEvent::set_event_handler(Some(Arc::new({
        let proxy = proxy.clone();
        move |event| {
            let _ = proxy.send_event(UserEvent::Menu(event));
        }
    })));
    let daemon = Daemon::new(move |tooltip| {
        let _ = proxy.send_event(UserEvent::Tooltip(tooltip));
    });

    let quit = MenuItem::new("Quit", true, None);
    let menu = Menu::with_items(&[&quit]).expect("tray menu");
    let mut tray = None;

    event_loop.run(move |event, _, flow| {
        *flow = ControlFlow::Wait;
        match event {
            // macOS needs the icon made once the loop runs.
            Event::NewEvents(StartCause::Init) => {
                let png = icon::TRAY[usize::from(cfg!(target_os = "macos"))];
                let (rgba, width, height) = icon::decode(png);
                tray = Some(
                    TrayIconBuilder::new()
                        .with_icon(Icon::from_rgba(rgba, width, height).expect("tray icon"))
                        .with_tooltip(IDLE)
                        .with_menu(Box::new(menu.clone()))
                        .with_menu_on_left_click(false)
                        .build()
                        .expect("tray icon"),
                );
                daemon.start(false);
            }
            Event::UserEvent(UserEvent::Tray(TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            })) => daemon.start(true),
            Event::UserEvent(UserEvent::Menu(event)) if event.id == *quit.id() => {
                daemon.stop();
                // Without this the icon lingers in the tray until hovered.
                tray = None;
                *flow = ControlFlow::Exit;
            }
            Event::UserEvent(UserEvent::Tooltip(tooltip)) => {
                if let Some(tray) = &tray {
                    let _ = tray.set_tooltip(Some(tooltip));
                }
            }
            _ => {}
        }
    })
}
