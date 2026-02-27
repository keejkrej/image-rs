#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    let result = if std::env::args_os().count() <= 1 {
        image_rs::ui::run(None)
    } else {
        image_rs::run_cli()
    };

    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

