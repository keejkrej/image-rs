use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use eframe::egui;
use serde_json::{Map, Value, json};

use super::command_registry;
use super::state::installed_macros_dir;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct MacroCommandInvocation {
    pub(super) command_id: String,
    pub(super) params: Option<Value>,
}

#[derive(Debug, Default)]
pub(super) struct MacroRunReport {
    pub(super) executed: usize,
    pub(super) blocked: usize,
    pub(super) unknown: usize,
    pub(super) unimplemented: usize,
    pub(super) lines: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct NamedMacroBlock {
    pub(super) name: String,
    pub(super) shortcut: Option<String>,
    pub(super) start_line: usize,
    pub(super) statements: Vec<(usize, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct InstalledMacroMenuEntry {
    pub(super) path: PathBuf,
    pub(super) macro_name: String,
    pub(super) label: String,
    pub(super) shortcut: Option<String>,
    pub(super) submenu: Option<String>,
}

pub(super) fn parse_macro_command_line(
    raw_line: &str,
    catalog: &command_registry::CommandCatalog,
) -> Result<Option<MacroCommandInvocation>, String> {
    let stripped = strip_macro_line_comment(raw_line);
    let line = stripped.trim();
    if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
        return Ok(None);
    }
    if is_ignored_macro_call(line) {
        return Ok(None);
    }

    if line.trim_end_matches(';').trim() == "close" {
        return Ok(Some(MacroCommandInvocation {
            command_id: "file.close".to_string(),
            params: None,
        }));
    }

    if let Some((option, state)) = parse_imagej_set_option_macro_call(line)? {
        return Ok(Some(MacroCommandInvocation {
            command_id: "macro.set_option".to_string(),
            params: Some(json!({
                "option": option,
                "state": state,
            })),
        }));
    }

    if let Some(target) = parse_imagej_call_macro_call(line)? {
        return Ok(Some(MacroCommandInvocation {
            command_id: "macro.call".to_string(),
            params: Some(json!({ "target": target })),
        }));
    }

    if let Some(invocation) = parse_imagej_namespace_command_macro_call(line) {
        return Ok(Some(invocation));
    }

    if let Some(target) = parse_imagej_builtin_macro_call(line) {
        return Ok(Some(MacroCommandInvocation {
            command_id: "macro.builtin_call".to_string(),
            params: Some(json!({ "target": target })),
        }));
    }

    if let Some(invocation) = parse_imagej_simple_builtin_macro_call(line) {
        return Ok(Some(invocation));
    }

    if let Some((label, options)) = parse_imagej_run_macro_call(line)? {
        let command_id =
            resolve_macro_command_id(&label, catalog).unwrap_or_else(|| label.to_string());
        return Ok(Some(MacroCommandInvocation {
            command_id,
            params: options.map(|options| macro_options_to_json(&options)),
        }));
    }

    let (command_id, params) = if let Some((command_id, raw_params)) = line.split_once('|') {
        (command_id.trim(), Some(raw_params.trim()))
    } else {
        (line, None)
    };
    if command_id.is_empty() {
        return Ok(None);
    }

    let params = match params {
        Some(raw) => Some(
            serde_json::from_str(raw).map_err(|error| format!("invalid params JSON ({error})"))?,
        ),
        None => None,
    };

    Ok(Some(MacroCommandInvocation {
        command_id: command_id.to_string(),
        params,
    }))
}

pub(super) fn strip_macro_line_comment(line: &str) -> String {
    let mut output = String::new();
    let mut string_delimiter: Option<char> = None;
    let mut escaped = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if escaped {
            output.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' if string_delimiter.is_some() => {
                output.push(ch);
                escaped = true;
            }
            '"' | '\'' if string_delimiter == Some(ch) => {
                output.push(ch);
                string_delimiter = None;
            }
            '"' | '\'' if string_delimiter.is_none() => {
                output.push(ch);
                string_delimiter = Some(ch);
            }
            '/' if string_delimiter.is_none() && chars.peek() == Some(&'/') => break,
            _ => output.push(ch),
        }
    }
    output
}

fn is_ignored_macro_call(line: &str) -> bool {
    let line = line.trim_end_matches(';').trim();
    ["requires", "setBatchMode"].iter().any(|name| {
        line.strip_prefix(name)
            .is_some_and(|rest| rest.trim_start().starts_with('('))
    })
}

pub(super) fn macro_source_executable_lines(contents: &str) -> Vec<(usize, String)> {
    let mut output = Vec::new();
    for (index, raw_line) in contents.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with('#')
            || line == "{"
            || line == "}"
        {
            continue;
        }

        if line.starts_with("macro ") {
            if let Some((_, rest)) = line.split_once('{') {
                let inline_body = rest.trim().trim_end_matches('}').trim();
                for statement in split_macro_statements(inline_body) {
                    output.push((index + 1, statement));
                }
            }
            continue;
        }

        if line.ends_with('{') {
            continue;
        }

        for statement in split_macro_statements(line) {
            output.push((index + 1, statement));
        }
    }
    output
}

pub(super) fn macro_source_named_blocks(contents: &str) -> Vec<NamedMacroBlock> {
    let mut blocks = Vec::new();
    let mut current: Option<NamedMacroBlock> = None;

    for (index, raw_line) in contents.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
            continue;
        }

        if let Some(block) = current.as_mut() {
            if let Some((before_close, _)) = line.split_once('}') {
                for statement in split_macro_statements(before_close) {
                    block.statements.push((line_number, statement));
                }
                if let Some(block) = current.take() {
                    blocks.push(block);
                }
            } else {
                for statement in split_macro_statements(line) {
                    block.statements.push((line_number, statement));
                }
            }
            continue;
        }

        let Some((name, inline_body)) = parse_macro_declaration(line) else {
            continue;
        };
        let mut block = NamedMacroBlock {
            shortcut: macro_name_shortcut(&name),
            name,
            start_line: line_number,
            statements: Vec::new(),
        };
        if let Some(body) = inline_body {
            let before_close = body
                .split_once('}')
                .map(|(before, _)| before)
                .unwrap_or(body);
            for statement in split_macro_statements(before_close) {
                block.statements.push((line_number, statement));
            }
            blocks.push(block);
        } else {
            current = Some(block);
        }
    }

    if let Some(block) = current
        && !block.statements.is_empty()
    {
        blocks.push(block);
    }

    blocks
}

fn parse_macro_declaration(line: &str) -> Option<(String, Option<&str>)> {
    if !line.starts_with("macro ") {
        return None;
    }
    let quote_start = find_macro_string_start(line)?;
    let (name, next) = parse_macro_string_literal(line, quote_start).ok()?;
    let rest = line[next..].trim_start();
    let inline_body = rest
        .strip_prefix('{')
        .map(str::trim)
        .filter(|body| !body.is_empty());
    Some((name, inline_body))
}

fn find_macro_string_start(input: &str) -> Option<usize> {
    let double_quote = input.find('"');
    let single_quote = input.find('\'');
    match (double_quote, single_quote) {
        (Some(double_quote), Some(single_quote)) => Some(double_quote.min(single_quote)),
        (Some(quote), None) | (None, Some(quote)) => Some(quote),
        (None, None) => None,
    }
}

pub(super) fn macro_name_shortcut(name: &str) -> Option<String> {
    let start = name.find('[')?;
    let end = name.rfind(']')?;
    if end <= start + 1 {
        return None;
    }
    let raw = &name[start + 1..end];
    let shortcut = if raw.len() > 1 {
        raw.to_ascii_uppercase()
    } else {
        raw.to_string()
    };
    if shortcut.len() > 3 {
        return None;
    }
    if !macro_shortcut_is_valid(&shortcut) {
        return None;
    }
    Some(shortcut)
}

fn macro_shortcut_is_valid(shortcut: &str) -> bool {
    if shortcut.chars().count() == 1 {
        return shortcut
            .chars()
            .next()
            .map(|shortcut| shortcut.is_ascii_alphanumeric())
            .unwrap_or(false);
    }
    if let Some(number) = shortcut.strip_prefix('F') {
        return number
            .parse::<u8>()
            .map(|number| (1..=12).contains(&number) && shortcut == format!("F{number}"))
            .unwrap_or(false);
    }
    if let Some(key) = shortcut.strip_prefix('N') {
        return macro_numpad_key_is_valid(key);
    }
    if let Some(key) = shortcut.strip_prefix('&') {
        return key.chars().count() == 1
            && key
                .chars()
                .next()
                .map(|key| key.is_ascii_alphanumeric())
                .unwrap_or(false);
    }
    false
}

fn macro_numpad_key_is_valid(key: &str) -> bool {
    key.chars().count() == 1
        && key
            .chars()
            .next()
            .map(|key| key.is_ascii_digit() || matches!(key, '/' | '*' | '-' | '+' | '.'))
            .unwrap_or(false)
}

pub(super) fn macro_display_name(name: &str) -> String {
    let Some(start) = name.find('[') else {
        return name.to_string();
    };
    let Some(end) = name.rfind(']') else {
        return name.to_string();
    };
    if end <= start {
        return name.to_string();
    }
    let mut display = String::new();
    display.push_str(name[..start].trim_end());
    display.push_str(name[end + 1..].trim_start());
    let display = display.trim().to_string();
    if display.is_empty() {
        name.to_string()
    } else {
        display
    }
}

pub(super) fn macro_shortcut_matches_text(shortcut: &str, text: &str) -> bool {
    if shortcut.chars().count() == 1 {
        return shortcut.eq_ignore_ascii_case(text);
    }
    if let Some(key) = shortcut.strip_prefix('N') {
        return key == text;
    }
    if let Some(key) = shortcut.strip_prefix('&') {
        return key.eq_ignore_ascii_case(text);
    }
    false
}

pub(super) fn function_key_for_macro_shortcut(shortcut: &str) -> Option<egui::Key> {
    match shortcut.to_ascii_uppercase().as_str() {
        "F1" => Some(egui::Key::F1),
        "F2" => Some(egui::Key::F2),
        "F3" => Some(egui::Key::F3),
        "F4" => Some(egui::Key::F4),
        "F5" => Some(egui::Key::F5),
        "F6" => Some(egui::Key::F6),
        "F7" => Some(egui::Key::F7),
        "F8" => Some(egui::Key::F8),
        "F9" => Some(egui::Key::F9),
        "F10" => Some(egui::Key::F10),
        "F11" => Some(egui::Key::F11),
        "F12" => Some(egui::Key::F12),
        _ => None,
    }
}

pub(super) fn macro_shortcut_matches_function_key(
    shortcut: &str,
    input: &egui::InputState,
) -> bool {
    function_key_for_macro_shortcut(shortcut)
        .map(|key| input.key_pressed(key))
        .unwrap_or(false)
}

fn split_macro_statements(line: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut current = String::new();
    let mut string_delimiter: Option<char> = None;
    let mut escaped = false;
    let mut bracket_depth = 0usize;

    for ch in line.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' if string_delimiter.is_some() => {
                current.push(ch);
                escaped = true;
            }
            '"' | '\'' if string_delimiter == Some(ch) => {
                current.push(ch);
                string_delimiter = None;
            }
            '"' | '\'' if string_delimiter.is_none() => {
                current.push(ch);
                string_delimiter = Some(ch);
            }
            '[' if string_delimiter.is_none() => {
                bracket_depth = bracket_depth.saturating_add(1);
                current.push(ch);
            }
            ']' if string_delimiter.is_none() => {
                bracket_depth = bracket_depth.saturating_sub(1);
                current.push(ch);
            }
            ';' if string_delimiter.is_none() && bracket_depth == 0 => {
                let statement = current.trim();
                if !statement.is_empty() {
                    statements.push(statement.to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    let statement = current.trim();
    if !statement.is_empty() && statement != "}" {
        statements.push(statement.trim_end_matches('}').trim().to_string());
    }

    statements
}

fn parse_imagej_run_macro_call(line: &str) -> Result<Option<(String, Option<String>)>, String> {
    let line = line.trim_end_matches(';').trim();
    let Some(rest) = line.strip_prefix("run") else {
        return Ok(None);
    };
    let rest = rest.trim_start();
    if !rest.starts_with('(') {
        return Ok(None);
    }
    let Some(inner) = rest
        .strip_prefix('(')
        .and_then(|rest| rest.strip_suffix(')'))
    else {
        return Err("malformed run(...) macro call".to_string());
    };

    let inner = inner.trim();
    let (label, next) = parse_macro_string_literal(inner, 0)?;
    let rest = inner[next..].trim_start();
    if rest.is_empty() {
        return Ok(Some((label, None)));
    }
    let Some(rest) = rest.strip_prefix(',') else {
        return Err("expected comma after macro command label".to_string());
    };
    let rest = rest.trim_start();
    let (options, next) = parse_macro_string_literal(rest, 0)?;
    if !rest[next..].trim().is_empty() {
        return Err("unexpected text after macro options".to_string());
    }

    Ok(Some((label, Some(options))))
}

fn parse_imagej_set_option_macro_call(line: &str) -> Result<Option<(String, bool)>, String> {
    let line = line.trim_end_matches(';').trim();
    let Some(rest) = line.strip_prefix("setOption") else {
        return Ok(None);
    };
    let rest = rest.trim_start();
    if !rest.starts_with('(') {
        return Ok(None);
    }
    let Some(inner) = rest
        .strip_prefix('(')
        .and_then(|rest| rest.strip_suffix(')'))
    else {
        return Err("malformed setOption(...) macro call".to_string());
    };

    let inner = inner.trim();
    let (option, next) = parse_macro_string_literal(inner, 0)?;
    let rest = inner[next..].trim_start();
    if rest.is_empty() {
        return Ok(Some((option, true)));
    }
    let Some(rest) = rest.strip_prefix(',') else {
        return Err("expected comma after setOption name".to_string());
    };
    let rest = rest.trim();
    let state = match rest {
        "true" => true,
        "false" => false,
        "1" => true,
        "0" => false,
        _ => return Err("setOption state must be true or false".to_string()),
    };
    Ok(Some((option, state)))
}

fn parse_imagej_call_macro_call(line: &str) -> Result<Option<String>, String> {
    let line = line.trim_end_matches(';').trim();
    let Some(call_start) = line.find("call") else {
        return Ok(None);
    };
    let rest = line[call_start + "call".len()..].trim_start();
    if !rest.starts_with('(') {
        return Ok(None);
    }
    let Some(inner) = rest
        .strip_prefix('(')
        .and_then(|rest| rest.strip_suffix(')'))
    else {
        return Err("malformed call(...) macro call".to_string());
    };
    let inner = inner.trim();
    let (target, _) = parse_macro_string_literal(inner, 0)?;
    Ok(Some(target))
}

fn parse_imagej_builtin_macro_call(line: &str) -> Option<String> {
    let line = line.trim_end_matches(';').trim();
    for prefix in ["Dialog.", "Overlay.", "Roi.", "Stack.", "Property."] {
        if let Some(rest) = line.strip_prefix(prefix) {
            let name = rest
                .split(|ch: char| ch == '(' || ch.is_whitespace())
                .next()
                .unwrap_or_default();
            if !name.is_empty() {
                return Some(format!("{prefix}{name}"));
            }
        }
    }
    None
}

fn parse_imagej_namespace_command_macro_call(line: &str) -> Option<MacroCommandInvocation> {
    let line = line.trim_end_matches(';').trim();
    match line {
        "Overlay.show" | "Overlay.show()" => {
            return Some(MacroCommandInvocation {
                command_id: "image.overlay.show".to_string(),
                params: None,
            });
        }
        "Overlay.hide" | "Overlay.hide()" => {
            return Some(MacroCommandInvocation {
                command_id: "image.overlay.hide".to_string(),
                params: None,
            });
        }
        _ => {}
    }

    let open = line.find('(')?;
    let target = line[..open].trim();
    let inner = line[open + 1..].strip_suffix(')')?.trim();

    if target == "Stack.setSlice"
        && let Ok(slice) = inner.parse::<usize>()
    {
        return Some(MacroCommandInvocation {
            command_id: "image.stacks.set".to_string(),
            params: Some(json!({ "slice": slice })),
        });
    }

    if target == "Roi.setStrokeWidth"
        && let Some(args) = parse_macro_number_args(inner, 1)
    {
        return Some(MacroCommandInvocation {
            command_id: "edit.options.line_width".to_string(),
            params: Some(json!({ "width": args[0] })),
        });
    }

    if target == "Roi.setName"
        && let Ok((name, index)) = parse_macro_string_literal(inner, 0)
        && inner[index..].trim().is_empty()
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.set_roi_name".to_string(),
            params: Some(json!({ "name": name })),
        });
    }

    if target == "Property.set"
        && let Some((key, value)) = parse_macro_two_string_args(inner)
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.set_metadata".to_string(),
            params: Some(json!({ "key": key, "value": value })),
        });
    }

    if target == "Overlay.addSelection" {
        return Some(MacroCommandInvocation {
            command_id: "image.overlay.add_selection".to_string(),
            params: None,
        });
    }

    if target == "Overlay.removeRois"
        && let Ok((name, index)) = parse_macro_string_literal(inner, 0)
        && inner[index..].trim().is_empty()
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.remove_overlay_rois".to_string(),
            params: Some(json!({ "name": name })),
        });
    }

    if target == "Overlay.removeSelection"
        && let Ok(index) = inner.parse::<usize>()
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.remove_overlay_selection".to_string(),
            params: Some(json!({ "index": index })),
        });
    }

    if target == "Overlay.activateSelection"
        && let Ok(index) = inner.parse::<usize>()
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.activate_overlay_selection".to_string(),
            params: Some(json!({ "index": index })),
        });
    }

    None
}

fn parse_imagej_simple_builtin_macro_call(line: &str) -> Option<MacroCommandInvocation> {
    let line = line.trim_end_matches(';').trim();
    let open = line.find('(')?;
    let name = line[..open].trim();
    if name
        .chars()
        .any(|ch| !(ch.is_ascii_alphanumeric() || ch == '_'))
    {
        return None;
    }
    let inner = line[open + 1..].strip_suffix(')')?.trim();

    if name == "selectNone" {
        return Some(MacroCommandInvocation {
            command_id: "edit.selection.none".to_string(),
            params: None,
        });
    }

    if name == "setSlice"
        && let Ok(slice) = inner.parse::<usize>()
    {
        return Some(MacroCommandInvocation {
            command_id: "image.stacks.set".to_string(),
            params: Some(json!({ "slice": slice })),
        });
    }

    if name == "selectImage"
        && let Ok(id) = inner.parse::<u64>()
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.select_image".to_string(),
            params: Some(json!({ "id": id })),
        });
    }

    if name == "setTool"
        && let Some(params) = parse_macro_set_tool_args(inner)
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.set_tool".to_string(),
            params: Some(params),
        });
    }

    if name == "setPasteMode"
        && let Ok((mode, index)) = parse_macro_string_literal(inner, 0)
        && inner[index..].trim().is_empty()
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.set_paste_mode".to_string(),
            params: Some(json!({ "mode": mode })),
        });
    }

    if matches!(name, "makeRectangle" | "makeOval")
        && let Some(args) = parse_macro_number_args(inner, 4)
    {
        let command_id = if name == "makeRectangle" {
            "macro.make_rectangle"
        } else {
            "macro.make_oval"
        };
        return Some(MacroCommandInvocation {
            command_id: command_id.to_string(),
            params: Some(json!({
                "x": args[0],
                "y": args[1],
                "width": args[2],
                "height": args[3],
            })),
        });
    }

    if name == "makeLine"
        && let Some(points) = parse_macro_point_list_args(inner, 2)
    {
        if points.len() == 2 {
            return Some(MacroCommandInvocation {
                command_id: "macro.make_line".to_string(),
                params: Some(json!({
                    "x1": points[0].x,
                    "y1": points[0].y,
                    "x2": points[1].x,
                    "y2": points[1].y,
                })),
            });
        }
        return Some(MacroCommandInvocation {
            command_id: "macro.make_selection".to_string(),
            params: Some(json!({
                "selection_type": "polyline",
                "points": points
                    .into_iter()
                    .map(|point| json!({"x": point.x, "y": point.y}))
                    .collect::<Vec<_>>(),
            })),
        });
    }

    if name == "makePolygon"
        && let Some(points) = parse_macro_point_list_args(inner, 2)
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.make_selection".to_string(),
            params: Some(json!({
                "selection_type": "polygon",
                "points": points
                    .into_iter()
                    .map(|point| json!({"x": point.x, "y": point.y}))
                    .collect::<Vec<_>>(),
            })),
        });
    }

    if name == "makePoint"
        && let Some(points) = parse_macro_point_list_args(inner, 1)
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.make_selection".to_string(),
            params: Some(json!({
                "selection_type": "point",
                "points": points
                    .into_iter()
                    .map(|point| json!({"x": point.x, "y": point.y}))
                    .collect::<Vec<_>>(),
            })),
        });
    }

    if name == "makeSelection"
        && let Some((selection_type, points)) = parse_macro_make_selection_args(inner)
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.make_selection".to_string(),
            params: Some(json!({
                "selection_type": selection_type,
                "points": points
                    .into_iter()
                    .map(|point| json!({"x": point.x, "y": point.y}))
                    .collect::<Vec<_>>(),
            })),
        });
    }

    if name == "newImage"
        && let Some(params) = parse_macro_new_image_args(inner)
    {
        return Some(MacroCommandInvocation {
            command_id: "file.new".to_string(),
            params: Some(params),
        });
    }

    if name == "selectWindow"
        && let Ok((title, index)) = parse_macro_string_literal(inner, 0)
        && inner[index..].trim().is_empty()
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.select_window".to_string(),
            params: Some(json!({ "title": title })),
        });
    }

    if matches!(name, "setForegroundColor" | "setBackgroundColor")
        && let Some([red, green, blue]) = parse_macro_color_args(inner)
    {
        let target = if name == "setForegroundColor" {
            "foreground"
        } else {
            "background"
        };
        return Some(MacroCommandInvocation {
            command_id: "macro.set_color".to_string(),
            params: Some(json!({
                "target": target,
                "red": red,
                "green": green,
                "blue": blue,
            })),
        });
    }

    if name == "rename"
        && let Ok((title, index)) = parse_macro_string_literal(inner, 0)
        && inner[index..].trim().is_empty()
    {
        return Some(MacroCommandInvocation {
            command_id: "image.rename".to_string(),
            params: Some(json!({ "title": title })),
        });
    }

    if name == "setSelectionName"
        && let Ok((name, index)) = parse_macro_string_literal(inner, 0)
        && inner[index..].trim().is_empty()
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.set_roi_name".to_string(),
            params: Some(json!({ "name": name })),
        });
    }

    if name == "setMetadata"
        && let Some((key, value)) = parse_macro_two_string_args(inner)
    {
        return Some(MacroCommandInvocation {
            command_id: "macro.set_metadata".to_string(),
            params: Some(json!({ "key": key, "value": value })),
        });
    }

    if name == "open"
        && let Ok((path, index)) = parse_macro_string_literal(inner, 0)
        && inner[index..].trim().is_empty()
    {
        let (command_id, params) = if path.starts_with("http://") || path.starts_with("https://") {
            ("file.import.url", json!({ "url": path }))
        } else {
            ("file.open", json!({ "path": path }))
        };
        return Some(MacroCommandInvocation {
            command_id: command_id.to_string(),
            params: Some(params),
        });
    }

    if name == "save"
        && let Ok((path, index)) = parse_macro_string_literal(inner, 0)
        && inner[index..].trim().is_empty()
    {
        return Some(MacroCommandInvocation {
            command_id: "file.save_as".to_string(),
            params: Some(json!({ "path": path })),
        });
    }

    if name == "saveAs"
        && let Some((format, path)) = parse_macro_two_string_args(inner)
    {
        return Some(MacroCommandInvocation {
            command_id: "file.save_as".to_string(),
            params: Some(json!({ "format": format, "path": path })),
        });
    }

    if name == "close" {
        if inner.is_empty() {
            return Some(MacroCommandInvocation {
                command_id: "file.close".to_string(),
                params: None,
            });
        }
        if let Ok((title, index)) = parse_macro_string_literal(inner, 0)
            && inner[index..].trim().is_empty()
        {
            return Some(MacroCommandInvocation {
                command_id: "macro.close_window".to_string(),
                params: Some(json!({ "title": title })),
            });
        }
    }

    let acknowledged = [
        "setSlice",
        "selectImage",
        "selectWindow",
        "setForegroundColor",
        "setBackgroundColor",
        "setTool",
        "rename",
        "open",
        "save",
        "saveAs",
        "close",
        "setSelectionName",
        "setMetadata",
        "newImage",
        "makeRectangle",
        "makeOval",
        "makeLine",
        "makePolygon",
        "makePoint",
        "makeSelection",
        "setPasteMode",
        "print",
        "wait",
        "showStatus",
        "showMessage",
        "exit",
        "roiManager",
    ];
    if acknowledged.contains(&name) {
        return Some(MacroCommandInvocation {
            command_id: "macro.builtin_call".to_string(),
            params: Some(json!({ "target": name })),
        });
    }

    None
}

fn parse_macro_number_args(inner: &str, expected: usize) -> Option<Vec<f32>> {
    let args = inner
        .split(',')
        .map(str::trim)
        .map(|arg| arg.parse::<f32>().ok())
        .collect::<Option<Vec<_>>>()?;
    (args.len() == expected && args.iter().all(|arg| arg.is_finite())).then_some(args)
}

fn parse_macro_color_args(inner: &str) -> Option<[u8; 3]> {
    let args = parse_macro_number_args(inner, 3)?;
    let mut colors = [0; 3];
    for (index, value) in args.iter().enumerate() {
        if !(0.0..=255.0).contains(value) {
            return None;
        }
        colors[index] = value.round() as u8;
    }
    Some(colors)
}

fn parse_macro_set_tool_args(inner: &str) -> Option<Value> {
    if let Ok((name, index)) = parse_macro_string_literal(inner, 0)
        && inner[index..].trim().is_empty()
    {
        return macro_tool_command_from_name(&name).map(|(tool, mode)| {
            json!({
                "tool": tool,
                "mode": mode,
            })
        });
    }

    let id = inner.parse::<usize>().ok()?;
    macro_tool_command_from_id(id).map(|(tool, mode)| {
        json!({
            "tool": tool,
            "mode": mode,
        })
    })
}

fn macro_tool_command_from_id(id: usize) -> Option<(&'static str, Option<&'static str>)> {
    match id {
        0 => Some((
            "launcher.tool.rect",
            Some("launcher.tool.rect.mode.rectangle"),
        )),
        1 => Some(("launcher.tool.oval", Some("launcher.tool.oval.mode.oval"))),
        2 => Some(("launcher.tool.poly", None)),
        3 => Some(("launcher.tool.free", None)),
        4 => Some((
            "launcher.tool.line",
            Some("launcher.tool.line.mode.straight"),
        )),
        5 => Some((
            "launcher.tool.line",
            Some("launcher.tool.line.mode.segmented"),
        )),
        6 => Some((
            "launcher.tool.line",
            Some("launcher.tool.line.mode.freehand"),
        )),
        7 => Some((
            "launcher.tool.point",
            Some("launcher.tool.point.mode.point"),
        )),
        8 => Some(("launcher.tool.wand", None)),
        9 => Some(("launcher.tool.text", None)),
        11 => Some(("launcher.tool.zoom", None)),
        12 => Some(("launcher.tool.hand", None)),
        13 => Some(("launcher.tool.dropper", None)),
        14 => Some(("launcher.tool.angle", None)),
        15 => Some(("launcher.tool.custom1", None)),
        16 => Some(("launcher.tool.custom2", None)),
        17 => Some(("launcher.tool.custom3", None)),
        _ => None,
    }
}

fn macro_tool_command_from_name(name: &str) -> Option<(&'static str, Option<&'static str>)> {
    let name = name.to_ascii_lowercase();
    if name.contains("round") {
        Some((
            "launcher.tool.rect",
            Some("launcher.tool.rect.mode.rounded"),
        ))
    } else if name.contains("rot") {
        Some((
            "launcher.tool.rect",
            Some("launcher.tool.rect.mode.rotated"),
        ))
    } else if name.contains("rect") {
        Some((
            "launcher.tool.rect",
            Some("launcher.tool.rect.mode.rectangle"),
        ))
    } else if name.contains("oval") {
        Some(("launcher.tool.oval", Some("launcher.tool.oval.mode.oval")))
    } else if name.contains("ellip") {
        Some((
            "launcher.tool.oval",
            Some("launcher.tool.oval.mode.ellipse"),
        ))
    } else if name.contains("brush") {
        Some(("launcher.tool.oval", Some("launcher.tool.oval.mode.brush")))
    } else if name.contains("polygon") {
        Some(("launcher.tool.poly", None))
    } else if name.contains("polyline") {
        Some((
            "launcher.tool.line",
            Some("launcher.tool.line.mode.segmented"),
        ))
    } else if name.contains("freeline") {
        Some((
            "launcher.tool.line",
            Some("launcher.tool.line.mode.freehand"),
        ))
    } else if name.contains("arrow") {
        Some(("launcher.tool.line", Some("launcher.tool.line.mode.arrow")))
    } else if name.contains("line") {
        Some((
            "launcher.tool.line",
            Some("launcher.tool.line.mode.straight"),
        ))
    } else if name.contains("free") {
        Some(("launcher.tool.free", None))
    } else if name.contains("multi") {
        Some((
            "launcher.tool.point",
            Some("launcher.tool.point.mode.multipoint"),
        ))
    } else if name.contains("point") {
        Some((
            "launcher.tool.point",
            Some("launcher.tool.point.mode.point"),
        ))
    } else if name.contains("wand") {
        Some(("launcher.tool.wand", None))
    } else if name.contains("text") {
        Some(("launcher.tool.text", None))
    } else if name.contains("hand") {
        Some(("launcher.tool.hand", None))
    } else if name.contains("zoom") || name.contains("magnifier") {
        Some(("launcher.tool.zoom", None))
    } else if name.contains("dropper") || name.contains("color") {
        Some(("launcher.tool.dropper", None))
    } else if name.contains("angle") {
        Some(("launcher.tool.angle", None))
    } else {
        None
    }
}

fn parse_macro_two_string_args(inner: &str) -> Option<(String, String)> {
    let (first, next) = parse_macro_string_literal(inner.trim(), 0).ok()?;
    let rest = inner.trim()[next..]
        .trim_start()
        .strip_prefix(',')?
        .trim_start();
    let (second, next) = parse_macro_string_literal(rest, 0).ok()?;
    rest[next..].trim().is_empty().then_some((first, second))
}

pub(super) fn macro_color_component(params: &Value, key: &str) -> Result<u8, String> {
    let Some(value) = params.get(key) else {
        return Err(format!("missing macro color component `{key}`"));
    };
    let Some(number) = value
        .as_u64()
        .map(|number| number as f64)
        .or_else(|| value.as_f64())
    else {
        return Err(format!("macro color component `{key}` must be numeric"));
    };
    if !number.is_finite() || !(0.0..=255.0).contains(&number) {
        return Err(format!(
            "macro color component `{key}` must be between 0 and 255"
        ));
    }
    Ok(number.round() as u8)
}

fn parse_macro_make_selection_args(inner: &str) -> Option<(String, Vec<egui::Pos2>)> {
    let (raw_type, rest) = inner.split_once(',')?;
    let (selection_type, _) = parse_macro_string_literal(raw_type.trim(), 0).ok()?;
    let min_points = if selection_type.to_ascii_lowercase().contains("point") {
        1
    } else {
        2
    };
    let points = parse_macro_point_list_args(rest, min_points)?;
    Some((selection_type, points))
}

fn parse_macro_point_list_args(inner: &str, min_points: usize) -> Option<Vec<egui::Pos2>> {
    let values = parse_macro_number_args(inner, inner.split(',').count())?;
    if values.len() < min_points * 2 || values.len() % 2 != 0 {
        return None;
    }
    Some(
        values
            .chunks_exact(2)
            .map(|pair| egui::pos2(pair[0], pair[1]))
            .collect::<Vec<_>>(),
    )
}

fn parse_macro_new_image_args(inner: &str) -> Option<Value> {
    let (title, next) = parse_macro_string_literal(inner.trim(), 0).ok()?;
    let rest = inner.trim()[next..]
        .trim_start()
        .strip_prefix(',')?
        .trim_start();
    let (image_type, next) = parse_macro_string_literal(rest, 0).ok()?;
    let rest = rest[next..].trim_start().strip_prefix(',')?.trim_start();
    let args = parse_macro_number_args(rest, 3)?;
    let pixel_type = if image_type.to_ascii_lowercase().contains("16-bit") {
        "u16"
    } else if image_type.to_ascii_lowercase().contains("8-bit")
        || image_type.to_ascii_lowercase().contains("rgb")
    {
        "u8"
    } else {
        "f32"
    };
    let channels = if image_type.to_ascii_lowercase().contains("rgb") {
        3
    } else {
        1
    };
    Some(json!({
        "title": title,
        "width": args[0].round().max(1.0) as usize,
        "height": args[1].round().max(1.0) as usize,
        "slices": args[2].round().max(1.0) as usize,
        "channels": channels,
        "frames": 1,
        "fill": if image_type.to_ascii_lowercase().contains("white") { 1.0 } else { 0.0 },
        "pixelType": pixel_type,
    }))
}

fn parse_macro_string_literal(input: &str, start: usize) -> Result<(String, usize), String> {
    let bytes = input.as_bytes();
    let Some(delimiter) = bytes
        .get(start)
        .copied()
        .filter(|byte| matches!(byte, b'"' | b'\''))
    else {
        return Err("expected macro string literal".to_string());
    };

    let mut output = String::new();
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            byte if byte == delimiter => return Ok((output, index + 1)),
            b'\\' => {
                index += 1;
                let Some(next) = bytes.get(index).copied() else {
                    return Err("unterminated macro string escape".to_string());
                };
                match next {
                    b'n' => output.push('\n'),
                    b'r' => output.push('\r'),
                    b't' => output.push('\t'),
                    b'"' => output.push('"'),
                    b'\'' => output.push('\''),
                    b'\\' => output.push('\\'),
                    other => output.push(other as char),
                }
            }
            other => output.push(other as char),
        }
        index += 1;
    }

    Err("unterminated macro string literal".to_string())
}

fn resolve_macro_command_id(
    label: &str,
    catalog: &command_registry::CommandCatalog,
) -> Option<String> {
    if catalog.entries.iter().any(|entry| entry.id == label) {
        return Some(label.to_string());
    }

    let normalized = normalize_macro_command_label(label);
    catalog
        .entries
        .iter()
        .find(|entry| normalize_macro_command_label(&entry.label) == normalized)
        .map(|entry| entry.id.clone())
}

pub(super) fn normalize_macro_command_label(label: &str) -> String {
    label
        .trim()
        .trim_end_matches('.')
        .trim_end_matches('\u{2026}')
        .replace('&', "")
        .to_ascii_lowercase()
}

pub(super) fn macro_options_to_json(options: &str) -> Value {
    let mut map = Map::new();
    for token in split_macro_option_tokens(options) {
        let (key, value) = token
            .split_once('=')
            .map(|(key, value)| (key.trim(), macro_option_value_to_json(value.trim())))
            .unwrap_or_else(|| (token.trim(), Value::Bool(true)));
        if !key.is_empty() {
            let key = macro_option_key_alias(key);
            map.insert(key.to_string(), value);
        }
    }
    Value::Object(map)
}

fn macro_option_key_alias(key: &str) -> &str {
    match key {
        "border" => "border_width",
        _ => key,
    }
}

fn split_macro_option_tokens(options: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut bracket_depth = 0usize;

    for ch in options.chars() {
        match ch {
            '[' => {
                bracket_depth = bracket_depth.saturating_add(1);
                current.push(ch);
            }
            ']' => {
                bracket_depth = bracket_depth.saturating_sub(1);
                current.push(ch);
            }
            ch if ch.is_whitespace() && bracket_depth == 0 => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn macro_option_value_to_json(value: &str) -> Value {
    let value = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(value);

    if value.eq_ignore_ascii_case("true") {
        return Value::Bool(true);
    }
    if value.eq_ignore_ascii_case("false") {
        return Value::Bool(false);
    }
    if let Ok(integer) = value.parse::<i64>() {
        return Value::from(integer);
    }
    if let Ok(float) = value.parse::<f64>() {
        return json!(float);
    }

    Value::String(value.to_string())
}

pub(super) fn macro_record_line_for_command(
    command_id: &str,
    params: Option<&Value>,
    catalog: &command_registry::CommandCatalog,
) -> Option<String> {
    let label = catalog
        .entries
        .iter()
        .find(|entry| entry.id == command_id)
        .map(|entry| entry.label.as_str())?;
    let label = escape_macro_string(label);
    let options = params.and_then(macro_params_to_options);

    Some(match options {
        Some(options) if !options.is_empty() => {
            format!("run(\"{label}\", \"{}\");", escape_macro_string(&options))
        }
        _ => format!("run(\"{label}\");"),
    })
}

fn macro_params_to_options(params: &Value) -> Option<String> {
    let Value::Object(map) = params else {
        return Some(format!("value={}", macro_value_to_option_string(params)));
    };
    if map.is_empty() {
        return None;
    }

    let mut keys = map.keys().collect::<Vec<_>>();
    keys.sort();
    let tokens = keys
        .into_iter()
        .filter_map(|key| {
            let value = map.get(key)?;
            if value.is_null() {
                return None;
            }
            if value.as_bool() == Some(true) {
                return Some(key.to_string());
            }
            Some(format!("{key}={}", macro_value_to_option_string(value)))
        })
        .collect::<Vec<_>>();

    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

fn macro_value_to_option_string(value: &Value) -> String {
    match value {
        Value::String(text) => bracket_macro_option_text(text),
        Value::Number(number) => number.to_string(),
        Value::Bool(value) => value.to_string(),
        other => bracket_macro_option_text(&other.to_string()),
    }
}

fn bracket_macro_option_text(text: &str) -> String {
    if text.is_empty()
        || text
            .chars()
            .any(|ch| ch.is_whitespace() || matches!(ch, '[' | ']' | '='))
    {
        format!("[{}]", text.replace(']', "\\]"))
    } else {
        text.to_string()
    }
}

fn escape_macro_string(text: &str) -> String {
    let mut output = String::new();
    for ch in text.chars() {
        match ch {
            '\\' => output.push_str("\\\\"),
            '"' => output.push_str("\\\""),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            other => output.push(other),
        }
    }
    output
}

pub(super) fn first_report_line(report: &str) -> &str {
    report.lines().next().unwrap_or(report)
}

pub(super) fn installed_macro_file_name(source: &Path) -> Result<String, String> {
    let file_name = source
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "macro path has no file name".to_string())?;
    let extension = source
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "ijm" | "txt") {
        return Err("only .ijm and .txt macro files can be installed".to_string());
    }

    if extension == "txt" && !file_name.contains('_') {
        let stem = source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| "macro path has no valid file stem".to_string())?;
        Ok(format!("{stem}.ijm"))
    } else {
        Ok(file_name.to_string())
    }
}

pub(super) fn install_macro_file_to_dir(
    source: &Path,
    install_dir: &Path,
) -> Result<PathBuf, String> {
    let installed_name = installed_macro_file_name(source)?;
    let contents = fs::read(source).map_err(|error| format!("macro read failed: {error}"))?;
    fs::create_dir_all(install_dir)
        .map_err(|error| format!("macro install directory create failed: {error}"))?;
    let target = install_dir.join(installed_name);
    fs::write(&target, contents).map_err(|error| format!("macro install write failed: {error}"))?;
    Ok(target)
}

pub(super) fn list_installed_macro_files_in_dir(install_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(install_dir) else {
        return Vec::new();
    };
    let mut macros = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| matches!(extension.to_ascii_lowercase().as_str(), "ijm" | "txt"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    macros.sort_by(|left, right| {
        left.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .cmp(
                right
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default(),
            )
    });
    macros
}

pub(super) fn installed_macro_blocks(path: &Path) -> Vec<NamedMacroBlock> {
    fs::read_to_string(path)
        .map(|contents| macro_source_named_blocks(&contents))
        .unwrap_or_default()
}

pub(super) fn startup_macro_set_path() -> Option<PathBuf> {
    let macros_dir = installed_macros_dir();
    for file_name in ["StartupMacros.txt", "StartupMacros.ijm"] {
        let path = macros_dir.join(file_name);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

pub(super) fn startup_auto_run_macro_block(source: &str) -> Option<NamedMacroBlock> {
    macro_source_named_blocks(source)
        .into_iter()
        .find(|block| block.name.starts_with("AutoRun"))
}

pub(super) fn macro_named_block_statement_map(
    source: &str,
) -> HashMap<String, Vec<(usize, String)>> {
    let mut blocks = HashMap::new();
    for block in macro_source_named_blocks(source) {
        let statements = block.statements.clone();
        blocks
            .entry(block.name.clone())
            .or_insert_with(|| statements.clone());

        let display_name = macro_display_name(&block.name);
        blocks
            .entry(display_name)
            .or_insert_with(|| statements.clone());

        let (_, menu_label) = macro_menu_label_parts(&block.name);
        let menu_display_name = macro_display_name(menu_label);
        blocks.entry(menu_display_name).or_insert(statements);
    }
    blocks
}

pub(super) fn installed_macro_menu_entry_from_block(
    path: &Path,
    block: &NamedMacroBlock,
) -> Option<InstalledMacroMenuEntry> {
    if block.name.starts_with("AutoRun")
        || block.name == "Popup Menu"
        || block.name.ends_with("Tool Selected")
        || block.name.contains("Tool Options")
    {
        return None;
    }

    let (submenu, raw_label) = macro_menu_label_parts(&block.name);
    Some(InstalledMacroMenuEntry {
        path: path.to_path_buf(),
        macro_name: block.name.clone(),
        label: macro_display_name(raw_label),
        shortcut: block.shortcut.clone(),
        submenu,
    })
}

fn macro_menu_label_parts(name: &str) -> (Option<String>, &str) {
    if name.starts_with('<')
        && let Some(separator) = name.find('>')
        && separator > 1
    {
        let submenu = name[1..separator].trim();
        let child = name[separator + 1..].trim();
        if !submenu.is_empty() && !child.is_empty() {
            return (Some(submenu.to_string()), child);
        }
    }
    (None, name)
}
