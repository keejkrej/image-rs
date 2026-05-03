use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LookupTable {
    Grays,
    Inverted,
    Fire,
    Ice,
    Spectrum,
    Rgb332,
    Red,
    Green,
    Blue,
    Cyan,
    Magenta,
    Yellow,
    RedGreen,
}

impl LookupTable {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Grays => "Grays",
            Self::Inverted => "Invert",
            Self::Fire => "Fire",
            Self::Ice => "Ice",
            Self::Spectrum => "Spectrum",
            Self::Rgb332 => "3-3-2 RGB",
            Self::Red => "Red",
            Self::Green => "Green",
            Self::Blue => "Blue",
            Self::Cyan => "Cyan",
            Self::Magenta => "Magenta",
            Self::Yellow => "Yellow",
            Self::RedGreen => "Red/Green",
        }
    }
}

pub(super) fn lookup_table_from_command(command_id: &str) -> Option<LookupTable> {
    Some(match command_id {
        "image.lookup.invert_lut" => LookupTable::Inverted,
        "image.color.invert_luts" => LookupTable::Inverted,
        "image.lookup.fire" => LookupTable::Fire,
        "image.lookup.grays" => LookupTable::Grays,
        "image.lookup.ice" => LookupTable::Ice,
        "image.lookup.spectrum" => LookupTable::Spectrum,
        "image.lookup.rgb332" => LookupTable::Rgb332,
        "image.lookup.red" => LookupTable::Red,
        "image.lookup.green" => LookupTable::Green,
        "image.lookup.blue" => LookupTable::Blue,
        "image.lookup.cyan" => LookupTable::Cyan,
        "image.lookup.magenta" => LookupTable::Magenta,
        "image.lookup.yellow" => LookupTable::Yellow,
        "image.lookup.red_green" => LookupTable::RedGreen,
        _ => return None,
    })
}

pub(super) fn lookup_table_color(lut: LookupTable, gray: u8) -> egui::Color32 {
    let g = gray;
    match lut {
        LookupTable::Grays => egui::Color32::from_rgb(g, g, g),
        LookupTable::Inverted => {
            let inv = 255 - g;
            egui::Color32::from_rgb(inv, inv, inv)
        }
        LookupTable::Red => egui::Color32::from_rgb(g, 0, 0),
        LookupTable::Green => egui::Color32::from_rgb(0, g, 0),
        LookupTable::Blue => egui::Color32::from_rgb(0, 0, g),
        LookupTable::Cyan => egui::Color32::from_rgb(0, g, g),
        LookupTable::Magenta => egui::Color32::from_rgb(g, 0, g),
        LookupTable::Yellow => egui::Color32::from_rgb(g, g, 0),
        LookupTable::RedGreen => egui::Color32::from_rgb(255 - g, g, 0),
        LookupTable::Rgb332 => {
            let r = (g & 0b1110_0000) | ((g & 0b1110_0000) >> 3) | ((g & 0b1110_0000) >> 6);
            let green = ((g & 0b0001_1100) << 3) | (g & 0b0001_1100) | ((g & 0b0001_1100) >> 3);
            let b = ((g & 0b0000_0011) << 6)
                | ((g & 0b0000_0011) << 4)
                | ((g & 0b0000_0011) << 2)
                | (g & 0b0000_0011);
            egui::Color32::from_rgb(r, green, b)
        }
        LookupTable::Fire => {
            let r = g.saturating_mul(2);
            let green = g.saturating_sub(64).saturating_mul(2);
            let b = g.saturating_sub(160).saturating_mul(3);
            egui::Color32::from_rgb(r, green, b)
        }
        LookupTable::Ice => {
            let r = g.saturating_sub(160).saturating_mul(3);
            let green = g.saturating_sub(32).saturating_mul(2);
            let b = 96u8.saturating_add(g / 2);
            egui::Color32::from_rgb(r, green, b)
        }
        LookupTable::Spectrum => {
            let segment = g as u16 * 6 / 256;
            let offset = ((g as u16 * 6) % 256) as u8;
            match segment {
                0 => egui::Color32::from_rgb(255, offset, 0),
                1 => egui::Color32::from_rgb(255 - offset, 255, 0),
                2 => egui::Color32::from_rgb(0, 255, offset),
                3 => egui::Color32::from_rgb(0, 255 - offset, 255),
                4 => egui::Color32::from_rgb(offset, 0, 255),
                _ => egui::Color32::from_rgb(255, 0, 255 - offset),
            }
        }
    }
}
