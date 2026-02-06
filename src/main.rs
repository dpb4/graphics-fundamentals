// #![windows_subsystem = "windows"]
use graphics_fundamentals::run;

pub fn main() {
    unsafe {
        std::env::set_var("WAYLAND_DISPLAY", ""); // Force X11 on Linux
    }
    run().unwrap();
}
