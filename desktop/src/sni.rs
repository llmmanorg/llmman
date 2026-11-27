//! Linux: a StatusNotifierItem over D-Bus, with no GTK.

use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};

use ksni::blocking::TrayMethods;
use ksni::menu::{MenuItem, StandardItem};

use crate::daemon::{Daemon, IDLE};
use crate::icon;

enum Event {
    Tooltip(&'static str),
    Quit,
}

struct Tray {
    daemon: Arc<Daemon>,
    events: Sender<Event>,
    tooltip: &'static str,
}

impl ksni::Tray for Tray {
    fn id(&self) -> String {
        "llmman-desktop".into()
    }

    fn title(&self) -> String {
        "llmman".into()
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        icon::TRAY
            .iter()
            .map(|png| {
                let (rgba, width, height) = icon::decode(png);
                ksni::Icon {
                    width: width as i32,
                    height: height as i32,
                    data: argb(&rgba),
                }
            })
            .collect()
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip {
            title: self.tooltip.into(),
            ..Default::default()
        }
    }

    fn activate(&mut self, _x: i32, _y: i32) {
        self.daemon.start(true);
    }

    fn menu(&self) -> Vec<MenuItem<Self>> {
        vec![
            StandardItem {
                label: "Quit".into(),
                activate: Box::new(|tray: &mut Tray| {
                    let _ = tray.events.send(Event::Quit);
                }),
                ..Default::default()
            }
            .into(),
        ]
    }
}

/// StatusNotifierItem pixmaps are ARGB32 in network byte order.
fn argb(rgba: &[u8]) -> Vec<u8> {
    let (pixels, _) = rgba.as_chunks::<4>();
    pixels
        .iter()
        .flat_map(|&[r, g, b, a]| [a, r, g, b])
        .collect()
}

/// Runs until Quit. Without a StatusNotifierWatcher (no tray host yet, say
/// at login) the icon appears once one does.
pub fn run() {
    let (events, inbox) = channel();
    let daemon = Daemon::new({
        let events = events.clone();
        move |tooltip| {
            let _ = events.send(Event::Tooltip(tooltip));
        }
    });
    let tray = Tray {
        daemon: daemon.clone(),
        events,
        tooltip: IDLE,
    };
    let handle = match tray.assume_sni_available(true).spawn() {
        Ok(handle) => handle,
        Err(e) => return eprintln!("llmman-desktop: no tray: {e}"),
    };
    daemon.start(false);
    for event in inbox {
        match event {
            Event::Tooltip(tooltip) => {
                handle.update(|tray| tray.tooltip = tooltip);
            }
            Event::Quit => break,
        }
    }
    daemon.stop();
    handle.shutdown().wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argb_moves_alpha_first() {
        assert_eq!(argb(&[1, 2, 3, 4, 5, 6, 7, 8]), [4, 1, 2, 3, 8, 5, 6, 7]);
    }
}
