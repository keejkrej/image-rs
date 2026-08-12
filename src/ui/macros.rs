#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::command_registry;

/// A toolkit-neutral command emitted by the ImageJ macro compatibility parser.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct MacroCommandInvocation {
    pub(super) command_id: String,
    pub(super) params: Option<Value>,
}

/// Parse one ImageJ macro statement into the command routing vocabulary used by the UI.
///
/// This intentionally handles the literal, command-oriented subset needed by ImageJ startup
/// macros and the recorder. Calls containing expressions are acknowledged through
/// `macro.builtin_call` so a future full interpreter can take them over without guessing.
pub(super) fn parse_macro_command_line(
    raw_statement: &str,
    catalog: &command_registry::CommandCatalog,
) -> Result<Option<MacroCommandInvocation>, String> {
    let stripped = strip_macro_line_comment(raw_statement);
    let statement = stripped.trim().trim_end_matches(';').trim();
    if statement.is_empty() || statement.starts_with('#') {
        return Ok(None);
    }
    if statement == "{"
        || statement == "}"
        || ((statement.starts_with("macro ") || statement.starts_with("function "))
            && statement.ends_with('{'))
    {
        return Ok(None);
    }
    if is_ignored_macro_call(statement) {
        return Ok(None);
    }
    if statement == "close" {
        return Ok(Some(invocation("file.close", None)));
    }

    let Some((name, arguments)) = parse_call(statement)? else {
        return parse_internal_command(statement).map(Some);
    };

    let parsed = match name {
        "run" => parse_run(arguments, catalog)?,
        "close" if arguments.trim().is_empty() => invocation("file.close", None),
        "close" => match parse_only_string(arguments) {
            Some(title) => invocation("macro.close_window", Some(json!({ "title": title }))),
            None => acknowledged_builtin(name),
        },
        "open" => match parse_only_string(arguments) {
            Some(path) if path.starts_with("http://") || path.starts_with("https://") => {
                invocation("file.import.url", Some(json!({ "url": path })))
            }
            Some(path) => invocation("file.open", Some(json!({ "path": path }))),
            None => acknowledged_builtin(name),
        },
        "save" => match parse_only_string(arguments) {
            Some(path) => invocation("file.save_as", Some(json!({ "path": path }))),
            None => acknowledged_builtin(name),
        },
        "saveAs" => match parse_two_string_args(arguments) {
            Some((format, path)) => invocation(
                "file.save_as",
                Some(json!({ "format": format, "path": path })),
            ),
            None => acknowledged_builtin(name),
        },
        "newImage" => match parse_new_image_args(arguments) {
            Some(params) => invocation("file.new", Some(params)),
            None => acknowledged_builtin(name),
        },
        "selectWindow" => match parse_only_string(arguments) {
            Some(title) => invocation("macro.select_window", Some(json!({ "title": title }))),
            None => acknowledged_builtin(name),
        },
        "selectImage" => match arguments.trim().parse::<u64>() {
            Ok(id) => invocation("macro.select_image", Some(json!({ "id": id }))),
            Err(_) => acknowledged_builtin(name),
        },
        "setTool" => match parse_tool_args(arguments) {
            Some(params) => invocation("macro.set_tool", Some(params)),
            None => acknowledged_builtin(name),
        },
        "makeRectangle" | "makeOval" => match parse_number_args(arguments) {
            Some(values) if values.len() == 4 => invocation(
                if name == "makeRectangle" {
                    "macro.make_rectangle"
                } else {
                    "macro.make_oval"
                },
                Some(json!({
                    "x": values[0],
                    "y": values[1],
                    "width": values[2],
                    "height": values[3],
                })),
            ),
            _ => acknowledged_builtin(name),
        },
        "makeLine" => match parse_points(arguments, 2) {
            Some(points) if points.len() == 2 => invocation(
                "macro.make_line",
                Some(json!({
                    "x1": points[0].0,
                    "y1": points[0].1,
                    "x2": points[1].0,
                    "y2": points[1].1,
                })),
            ),
            Some(points) => selection_invocation("polyline", points),
            None => acknowledged_builtin(name),
        },
        "makePolygon" => match parse_points(arguments, 2) {
            Some(points) => selection_invocation("polygon", points),
            None => acknowledged_builtin(name),
        },
        "makePoint" => match parse_points(arguments, 1) {
            Some(points) => selection_invocation("point", points),
            None => acknowledged_builtin(name),
        },
        "makeSelection" => match parse_selection_args(arguments) {
            Some((selection_type, points)) => selection_invocation(&selection_type, points),
            None => acknowledged_builtin(name),
        },
        "roiManager" => match parse_only_string(arguments) {
            Some(action) => invocation("macro.roi_manager", Some(json!({ "action": action }))),
            None => acknowledged_builtin(name),
        },
        "setMinAndMax" => match parse_number_args(arguments) {
            Some(values) if values.len() == 2 => invocation(
                "macro.set_min_and_max",
                Some(json!({
                    "minimum": values[0],
                    "maximum": values[1],
                })),
            ),
            _ => acknowledged_builtin(name),
        },
        "resetMinAndMax" if arguments.trim().is_empty() => {
            invocation("macro.reset_min_and_max", None)
        }
        "resetMinAndMax" => acknowledged_builtin(name),
        "setOption" => parse_set_option(arguments)?,
        _ => return Ok(Some(acknowledged_builtin(name))),
    };

    Ok(Some(parsed))
}

/// Parse every executable statement in a source buffer.
pub(super) fn parse_macro_source(
    source: &str,
    catalog: &command_registry::CommandCatalog,
) -> Result<Vec<MacroCommandInvocation>, String> {
    let mut invocations = Vec::new();
    for (index, statement) in split_macro_statements(source).into_iter().enumerate() {
        let Some(statement) = executable_statement_body(&statement) else {
            continue;
        };
        match parse_macro_command_line(statement, catalog) {
            Ok(Some(invocation)) => invocations.push(invocation),
            Ok(None) => {}
            Err(error) => return Err(format!("macro statement {}: {error}", index + 1)),
        }
    }
    Ok(invocations)
}

fn executable_statement_body(statement: &str) -> Option<&str> {
    let statement = statement.trim();
    if statement.starts_with("macro ") || statement.starts_with("function ") {
        let (_, body) = statement.split_once('{')?;
        let body = body.trim();
        return (!body.is_empty()).then_some(body);
    }
    Some(statement)
}

/// Split a macro source buffer on statement terminators without splitting quoted or bracketed
/// values. Both `//` and line-leading `#` comments are ignored outside string literals.
pub(super) fn split_macro_statements(source: &str) -> Vec<String> {
    let chars = source.chars().collect::<Vec<_>>();
    let mut statements = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;
    let mut line_comment = false;
    let mut bracket_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut line_has_code = false;
    let mut index = 0usize;

    while index < chars.len() {
        let ch = chars[index];
        if line_comment {
            if ch == '\n' {
                line_comment = false;
                finish_at_newline(
                    &mut current,
                    &mut statements,
                    quote,
                    bracket_depth,
                    paren_depth,
                );
                line_has_code = false;
            }
            index += 1;
            continue;
        }
        if escaped {
            current.push(ch);
            escaped = false;
            line_has_code = true;
            index += 1;
            continue;
        }
        if let Some(delimiter) = quote {
            current.push(ch);
            match ch {
                '\\' => escaped = true,
                value if value == delimiter => quote = None,
                _ => {}
            }
            line_has_code = true;
            index += 1;
            continue;
        }

        match ch {
            '/' if chars.get(index + 1) == Some(&'/') => {
                line_comment = true;
                index += 2;
                continue;
            }
            '#' if !line_has_code => {
                line_comment = true;
                index += 1;
                continue;
            }
            '\'' | '"' => {
                quote = Some(ch);
                current.push(ch);
                line_has_code = true;
            }
            '[' => {
                bracket_depth = bracket_depth.saturating_add(1);
                current.push(ch);
                line_has_code = true;
            }
            ']' => {
                bracket_depth = bracket_depth.saturating_sub(1);
                current.push(ch);
                line_has_code = true;
            }
            '(' => {
                paren_depth = paren_depth.saturating_add(1);
                current.push(ch);
                line_has_code = true;
            }
            ')' => {
                paren_depth = paren_depth.saturating_sub(1);
                current.push(ch);
                line_has_code = true;
            }
            ';' if bracket_depth == 0 && paren_depth == 0 => {
                push_statement(&mut current, &mut statements);
                line_has_code = false;
            }
            '\n' => {
                finish_at_newline(
                    &mut current,
                    &mut statements,
                    quote,
                    bracket_depth,
                    paren_depth,
                );
                line_has_code = false;
            }
            _ => {
                current.push(ch);
                line_has_code |= !ch.is_whitespace();
            }
        }
        index += 1;
    }
    push_statement(&mut current, &mut statements);
    statements
}

pub(super) fn strip_macro_line_comment(line: &str) -> String {
    let mut output = String::new();
    let mut quote = None;
    let mut escaped = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if escaped {
            output.push(ch);
            escaped = false;
            continue;
        }
        if let Some(delimiter) = quote {
            output.push(ch);
            match ch {
                '\\' => escaped = true,
                value if value == delimiter => quote = None,
                _ => {}
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                output.push(ch);
            }
            '/' if chars.peek() == Some(&'/') => break,
            _ => output.push(ch),
        }
    }
    output
}

pub(super) fn normalize_macro_command_label(label: &str) -> String {
    let mut normalized = label.trim();
    loop {
        let next = normalized
            .trim()
            .trim_end_matches('.')
            .trim_end_matches('\u{2026}')
            .trim();
        if next.len() == normalized.len() {
            break;
        }
        normalized = next;
    }
    normalized.replace('&', "").to_ascii_lowercase()
}

/// Parse ImageJ's whitespace-delimited option format into typed JSON values.
pub(super) fn macro_options_to_json(options: &str) -> Value {
    let mut map = Map::new();
    for token in split_macro_option_tokens(options) {
        let (key, value) = token
            .split_once('=')
            .map(|(key, value)| (key.trim(), macro_option_value_to_json(value.trim())))
            .unwrap_or_else(|| (token.trim(), Value::Bool(true)));
        if !key.is_empty() {
            map.insert(macro_option_key_alias(key).to_string(), value);
        }
    }
    Value::Object(map)
}

/// Produce an ImageJ `run(...)` line suitable for the macro recorder.
pub(super) fn macro_record_line_for_command(
    command_id: &str,
    params: Option<&Value>,
    catalog: &command_registry::CommandCatalog,
) -> Option<String> {
    match command_id {
        "macro.set_min_and_max" => {
            let params = params?;
            let minimum = params.get("minimum").and_then(Value::as_f64)?;
            let maximum = params.get("maximum").and_then(Value::as_f64)?;
            if !minimum.is_finite() || !maximum.is_finite() {
                return None;
            }
            return Some(format!("setMinAndMax({minimum}, {maximum});"));
        }
        "macro.reset_min_and_max" => return Some("resetMinAndMax();".to_string()),
        _ => {}
    }

    let label = catalog
        .entries
        .iter()
        .find(|entry| entry.id == command_id)
        .map(|entry| escape_macro_string(&entry.label))?;
    match params.and_then(macro_params_to_options) {
        Some(options) if !options.is_empty() => Some(format!(
            "run(\"{label}\", \"{}\");",
            escape_macro_string(&options)
        )),
        _ => Some(format!("run(\"{label}\");")),
    }
}

pub(super) fn startup_macro_path() -> PathBuf {
    image_rs_config_dir().join("RunAtStartup.ijm")
}

pub(super) fn installed_macros_dir() -> PathBuf {
    image_rs_config_dir().join("plugins").join("Macros")
}

pub(super) fn install_macro_file(source: &Path) -> Result<PathBuf, String> {
    install_macro_file_to_dir(source, &installed_macros_dir())
}

pub(super) fn install_macro_file_to_dir(
    source: &Path,
    install_dir: &Path,
) -> Result<PathBuf, String> {
    let installed_name = installed_macro_file_name(source)?;
    fs::create_dir_all(install_dir)
        .map_err(|error| format!("macro install directory create failed: {error}"))?;
    let target = install_dir.join(installed_name);
    fs::copy(source, &target).map_err(|error| format!("macro install copy failed: {error}"))?;
    Ok(target)
}

pub(super) fn installed_macro_file_name(source: &Path) -> Result<String, String> {
    let file_name = source
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "macro path has no file name".to_string())?;
    let extension = source
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
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

pub(super) fn list_installed_macro_files() -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(installed_macros_dir()) else {
        return Vec::new();
    };
    let mut files = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    matches!(extension.to_ascii_lowercase().as_str(), "ijm" | "txt")
                })
        })
        .collect::<Vec<_>>();
    files.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    files
}

fn invocation(command_id: &str, params: Option<Value>) -> MacroCommandInvocation {
    MacroCommandInvocation {
        command_id: command_id.to_string(),
        params,
    }
}

fn acknowledged_builtin(name: &str) -> MacroCommandInvocation {
    invocation(
        "macro.builtin_call",
        Some(json!({ "target": name.to_string() })),
    )
}

fn parse_run(
    arguments: &str,
    catalog: &command_registry::CommandCatalog,
) -> Result<MacroCommandInvocation, String> {
    let (label, next) = parse_macro_string_literal(arguments.trim(), 0)?;
    let remaining = arguments.trim()[next..].trim();
    let options = if remaining.is_empty() {
        None
    } else {
        let remaining = remaining
            .strip_prefix(',')
            .ok_or_else(|| "expected comma after macro command label".to_string())?
            .trim_start();
        let (options, next) = parse_macro_string_literal(remaining, 0)?;
        if !remaining[next..].trim().is_empty() {
            return Err("unexpected text after macro options".to_string());
        }
        Some(options)
    };
    let command_id = resolve_macro_command_id(&label, catalog).unwrap_or(label);
    Ok(invocation(
        &command_id,
        options.as_deref().map(macro_options_to_json),
    ))
}

fn parse_set_option(arguments: &str) -> Result<MacroCommandInvocation, String> {
    let input = arguments.trim();
    let (option, next) = parse_macro_string_literal(input, 0)?;
    let remaining = input[next..].trim();
    let state = if remaining.is_empty() {
        true
    } else {
        match remaining
            .strip_prefix(',')
            .ok_or_else(|| "expected comma after setOption name".to_string())?
            .trim()
        {
            "true" | "1" => true,
            "false" | "0" => false,
            _ => return Err("setOption state must be true or false".to_string()),
        }
    };
    Ok(invocation(
        "macro.set_option",
        Some(json!({ "option": option, "state": state })),
    ))
}

fn parse_call(statement: &str) -> Result<Option<(&str, &str)>, String> {
    let Some(open) = statement.find('(') else {
        return Ok(None);
    };
    let name = statement[..open].trim();
    if name.is_empty()
        || name
            .chars()
            .any(|character| !(character.is_ascii_alphanumeric() || character == '_'))
    {
        return Ok(None);
    }
    let rest = &statement[open + 1..];
    let Some(arguments) = rest.strip_suffix(')') else {
        return Err(format!("malformed {name}(...) macro call"));
    };
    Ok(Some((name, arguments)))
}

fn parse_internal_command(statement: &str) -> Result<MacroCommandInvocation, String> {
    let (command_id, params) = statement
        .split_once('|')
        .map(|(command, params)| (command.trim(), Some(params.trim())))
        .unwrap_or((statement.trim(), None));
    if command_id.is_empty() {
        return Err("macro command id is empty".to_string());
    }
    let params = params
        .map(|raw| {
            serde_json::from_str(raw).map_err(|error| format!("invalid params JSON ({error})"))
        })
        .transpose()?;
    Ok(invocation(command_id, params))
}

fn is_ignored_macro_call(statement: &str) -> bool {
    ["requires", "setBatchMode"].iter().any(|name| {
        statement
            .strip_prefix(name)
            .is_some_and(|rest| rest.trim_start().starts_with('('))
    })
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

fn parse_only_string(arguments: &str) -> Option<String> {
    let input = arguments.trim();
    let (value, next) = parse_macro_string_literal(input, 0).ok()?;
    input[next..].trim().is_empty().then_some(value)
}

fn parse_two_string_args(arguments: &str) -> Option<(String, String)> {
    let input = arguments.trim();
    let (first, next) = parse_macro_string_literal(input, 0).ok()?;
    let rest = input[next..].trim_start().strip_prefix(',')?.trim_start();
    let (second, next) = parse_macro_string_literal(rest, 0).ok()?;
    rest[next..].trim().is_empty().then_some((first, second))
}

fn parse_new_image_args(arguments: &str) -> Option<Value> {
    let input = arguments.trim();
    let (title, next) = parse_macro_string_literal(input, 0).ok()?;
    let rest = input[next..].trim_start().strip_prefix(',')?.trim_start();
    let (image_type, next) = parse_macro_string_literal(rest, 0).ok()?;
    let rest = rest[next..].trim_start().strip_prefix(',')?.trim_start();
    let dimensions = parse_number_args(rest)?;
    if dimensions.len() != 3 {
        return None;
    }
    let lower_type = image_type.to_ascii_lowercase();
    let pixel_type = if lower_type.contains("16-bit") {
        "u16"
    } else if lower_type.contains("8-bit") || lower_type.contains("rgb") {
        "u8"
    } else {
        "f32"
    };
    let channels = if lower_type.contains("rgb") { 3 } else { 1 };
    let fill = if lower_type.contains("white") {
        match pixel_type {
            "u8" => 255.0,
            "u16" => 65_535.0,
            _ => 1.0,
        }
    } else {
        0.0
    };
    Some(json!({
        "title": title,
        "width": dimensions[0].round().max(1.0) as usize,
        "height": dimensions[1].round().max(1.0) as usize,
        "slices": dimensions[2].round().max(1.0) as usize,
        "channels": channels,
        "frames": 1,
        "fill": fill,
        "pixelType": pixel_type,
    }))
}

fn parse_selection_args(arguments: &str) -> Option<(String, Vec<(f32, f32)>)> {
    let input = arguments.trim();
    let (selection_type, next) = parse_macro_string_literal(input, 0).ok()?;
    let rest = input[next..].trim_start().strip_prefix(',')?.trim_start();
    let min_points = if selection_type.to_ascii_lowercase().contains("point") {
        1
    } else {
        2
    };
    Some((selection_type, parse_points(rest, min_points)?))
}

fn selection_invocation(selection_type: &str, points: Vec<(f32, f32)>) -> MacroCommandInvocation {
    let points = points
        .into_iter()
        .map(|(x, y)| json!({ "x": x, "y": y }))
        .collect::<Vec<_>>();
    invocation(
        "macro.make_selection",
        Some(json!({ "selection_type": selection_type, "points": points })),
    )
}

fn parse_number_args(arguments: &str) -> Option<Vec<f32>> {
    let values = arguments
        .split(',')
        .map(str::trim)
        .map(|value| value.parse::<f32>().ok())
        .collect::<Option<Vec<_>>>()?;
    (!values.is_empty() && values.iter().all(|value| value.is_finite())).then_some(values)
}

fn parse_points(arguments: &str, minimum_points: usize) -> Option<Vec<(f32, f32)>> {
    let values = parse_number_args(arguments)?;
    if values.len() < minimum_points * 2 || values.len() % 2 != 0 {
        return None;
    }
    Some(
        values
            .chunks_exact(2)
            .map(|pair| (pair[0], pair[1]))
            .collect(),
    )
}

fn parse_tool_args(arguments: &str) -> Option<Value> {
    let input = arguments.trim();
    let mapping = if let Some(name) = parse_only_string(input) {
        tool_command_from_name(&name)
    } else {
        input.parse::<usize>().ok().and_then(tool_command_from_id)
    }?;
    Some(json!({ "tool": mapping.0, "mode": mapping.1 }))
}

fn tool_command_from_id(id: usize) -> Option<(&'static str, Option<&'static str>)> {
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

fn tool_command_from_name(name: &str) -> Option<(&'static str, Option<&'static str>)> {
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

fn parse_macro_string_literal(input: &str, start: usize) -> Result<(String, usize), String> {
    let Some(tail) = input.get(start..) else {
        return Err("expected macro string literal".to_string());
    };
    let mut chars = tail.char_indices();
    let Some((_, delimiter @ ('"' | '\''))) = chars.next() else {
        return Err("expected macro string literal".to_string());
    };
    let mut output = String::new();
    let mut escaped = false;
    for (offset, character) in chars {
        if escaped {
            output.push(match character {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '"' => '"',
                '\'' => '\'',
                '\\' => '\\',
                other => other,
            });
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == delimiter {
            return Ok((output, start + offset + character.len_utf8()));
        } else {
            output.push(character);
        }
    }
    if escaped {
        return Err("unterminated macro string escape".to_string());
    }
    Err("unterminated macro string literal".to_string())
}

fn split_macro_option_tokens(options: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut bracket_depth = 0usize;
    let mut quote = None;
    let mut escaped = false;
    for ch in options.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if let Some(delimiter) = quote {
            current.push(ch);
            match ch {
                '\\' => escaped = true,
                value if value == delimiter => quote = None,
                _ => {}
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                current.push(ch);
            }
            '\\' if bracket_depth > 0 => {
                current.push(ch);
                escaped = true;
            }
            '[' => {
                bracket_depth = bracket_depth.saturating_add(1);
                current.push(ch);
            }
            ']' => {
                bracket_depth = bracket_depth.saturating_sub(1);
                current.push(ch);
            }
            value if value.is_whitespace() && bracket_depth == 0 => {
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

fn macro_option_key_alias(key: &str) -> &str {
    match key {
        "border" => "border_width",
        _ => key,
    }
}

fn macro_option_value_to_json(raw: &str) -> Value {
    let value = raw
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .or_else(|| {
            raw.strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
        })
        .or_else(|| {
            raw.strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
        .unwrap_or(raw);
    if value.eq_ignore_ascii_case("true") {
        Value::Bool(true)
    } else if value.eq_ignore_ascii_case("false") {
        Value::Bool(false)
    } else if value.eq_ignore_ascii_case("null") {
        Value::Null
    } else if let Ok(integer) = value.parse::<i64>() {
        Value::from(integer)
    } else if let Ok(float) = value.parse::<f64>() {
        json!(float)
    } else {
        Value::String(value.replace("\\]", "]"))
    }
}

fn macro_params_to_options(params: &Value) -> Option<String> {
    let Value::Object(map) = params else {
        return Some(format!("value={}", macro_value_to_option_string(params)));
    };
    let mut keys = map.keys().collect::<Vec<_>>();
    keys.sort();
    let tokens = keys
        .into_iter()
        .filter_map(|key| {
            let value = map.get(key)?;
            if value.is_null() {
                None
            } else if value.as_bool() == Some(true) {
                Some(key.to_string())
            } else {
                Some(format!("{key}={}", macro_value_to_option_string(value)))
            }
        })
        .collect::<Vec<_>>();
    (!tokens.is_empty()).then(|| tokens.join(" "))
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
            .any(|character| character.is_whitespace() || matches!(character, '[' | ']' | '='))
    {
        format!("[{}]", text.replace(']', "\\]"))
    } else {
        text.to_string()
    }
}

fn escape_macro_string(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn image_rs_config_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("image-rs");
    }
    if let Some(path) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(path).join("image-rs");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".config").join("image-rs");
    }
    PathBuf::from(".").join("image-rs")
}

fn finish_at_newline(
    current: &mut String,
    statements: &mut Vec<String>,
    quote: Option<char>,
    bracket_depth: usize,
    paren_depth: usize,
) {
    if quote.is_none() && bracket_depth == 0 && paren_depth == 0 {
        push_statement(current, statements);
    } else {
        current.push('\n');
    }
}

fn push_statement(current: &mut String, statements: &mut Vec<String>) {
    let statement = current.trim();
    if !statement.is_empty() && statement != "{" && statement != "}" {
        statements.push(statement.to_string());
    }
    current.clear();
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use serde_json::json;
    use tempfile::tempdir;

    use super::*;

    fn parse(statement: &str) -> MacroCommandInvocation {
        parse_macro_command_line(statement, &command_registry::command_catalog())
            .expect("statement should parse")
            .expect("statement should emit a command")
    }

    #[test]
    fn run_resolves_menu_labels_and_types_options() {
        let invocation = parse(
            r#"run("Enhance Contrast...", "saturated=0.35 normalize title=[My Image] passes=2 modal=false");"#,
        );
        assert_eq!(invocation.command_id, "process.enhance_contrast");
        assert_eq!(
            invocation.params,
            Some(json!({
                "saturated": 0.35,
                "normalize": true,
                "title": "My Image",
                "passes": 2,
                "modal": false,
            }))
        );
    }

    #[test]
    fn apply_lut_scope_options_survive_parse_and_recorder_round_trip() {
        let catalog = command_registry::command_catalog();
        for (scope, params) in [
            ("slice", json!({ "slice": true })),
            ("stack", json!({ "stack": true })),
        ] {
            let expected = invocation("image.lookup.apply_lut", Some(params.clone()));
            assert_eq!(parse(&format!(r#"run("Apply LUT", "{scope}");"#)), expected);

            let recorded =
                macro_record_line_for_command("image.lookup.apply_lut", Some(&params), &catalog)
                    .expect("Apply LUT should be recordable");
            assert_eq!(recorded, format!(r#"run("Apply LUT", "{scope}");"#));
            assert_eq!(parse(&recorded), expected);
        }
    }

    #[test]
    fn option_parser_round_trips_escaped_closing_brackets() {
        assert_eq!(
            macro_options_to_json(r#"title=[A \] bracket] enabled"#),
            json!({ "title": "A ] bracket", "enabled": true })
        );
    }

    #[test]
    fn file_and_window_builtins_route_to_frontend_commands() {
        assert_eq!(parse("close();"), invocation("file.close", None));
        assert_eq!(
            parse(r#"open("/tmp/source.tif");"#),
            invocation("file.open", Some(json!({ "path": "/tmp/source.tif" })))
        );
        assert_eq!(
            parse(r#"open("https://example.com/a.tif");"#),
            invocation(
                "file.import.url",
                Some(json!({ "url": "https://example.com/a.tif" }))
            )
        );
        assert_eq!(
            parse(r#"saveAs("Tiff", "/tmp/output");"#),
            invocation(
                "file.save_as",
                Some(json!({ "format": "Tiff", "path": "/tmp/output" }))
            )
        );
        assert_eq!(
            parse(r#"selectWindow("cells");"#),
            invocation("macro.select_window", Some(json!({ "title": "cells" })))
        );
        assert_eq!(
            parse("selectImage(3);"),
            invocation("macro.select_image", Some(json!({ "id": 3 })))
        );
    }

    #[test]
    fn new_image_and_tools_keep_imagej_literals() {
        assert_eq!(
            parse(r#"newImage("luts", "RGB White", 256, 48, 1);"#),
            invocation(
                "file.new",
                Some(json!({
                    "title": "luts",
                    "width": 256,
                    "height": 48,
                    "slices": 1,
                    "channels": 3,
                    "frames": 1,
                    "fill": 255.0,
                    "pixelType": "u8",
                }))
            )
        );
        assert_eq!(
            parse(r#"setTool("arrow");"#).params,
            Some(json!({
                "tool": "launcher.tool.line",
                "mode": "launcher.tool.line.mode.arrow",
            }))
        );
        assert_eq!(
            parse("setTool(7);").params,
            Some(json!({
                "tool": "launcher.tool.point",
                "mode": "launcher.tool.point.mode.point",
            }))
        );
    }

    #[test]
    fn literal_roi_calls_emit_toolkit_neutral_points() {
        assert_eq!(
            parse("makeRectangle(0, 1, 256, 32);").params,
            Some(json!({ "x": 0.0, "y": 1.0, "width": 256.0, "height": 32.0 }))
        );
        assert_eq!(parse("makeOval(1, 2, 3, 4);").command_id, "macro.make_oval");
        assert_eq!(
            parse("makeLine(1, 2, 3, 4);").params,
            Some(json!({ "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0 }))
        );
        assert_eq!(
            parse(r#"makeSelection("freehand", 0, 0, 10, 0, 10, 10);"#).params,
            Some(json!({
                "selection_type": "freehand",
                "points": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 10.0, "y": 0.0},
                    {"x": 10.0, "y": 10.0},
                ],
            }))
        );
        assert_eq!(
            parse(r#"makeSelection("point", 4, 5);"#).params,
            Some(json!({
                "selection_type": "point",
                "points": [{"x": 4.0, "y": 5.0}],
            }))
        );
    }

    #[test]
    fn roi_manager_calls_keep_imagej_action_names() {
        assert_eq!(
            parse(r#"roiManager("Add");"#),
            invocation("macro.roi_manager", Some(json!({ "action": "Add" })))
        );
        assert_eq!(
            parse(r#"roiManager("Measure");"#),
            invocation("macro.roi_manager", Some(json!({ "action": "Measure" })))
        );
    }

    #[test]
    fn set_option_uses_acknowledgement_route() {
        assert_eq!(
            parse(r#"setOption("Stack position", false);"#),
            invocation(
                "macro.set_option",
                Some(json!({ "option": "Stack position", "state": false }))
            )
        );
    }

    #[test]
    fn display_range_builtins_emit_dedicated_invocations() {
        assert_eq!(
            parse("setMinAndMax(-12.5, 4095);"),
            invocation(
                "macro.set_min_and_max",
                Some(json!({ "minimum": -12.5, "maximum": 4095.0 }))
            )
        );
        assert_eq!(
            parse("resetMinAndMax();"),
            invocation("macro.reset_min_and_max", None)
        );
    }

    #[test]
    fn source_splitter_respects_quotes_brackets_multiline_calls_and_comments() {
        let source = r#"
            run("URL...", "url=http://example.com/a;b"); // do not split URL
            run(
                "Duplicate...",
                "title=[a; b]"
            ); # comment
            close(); run("Smooth");
        "#;
        let statements = split_macro_statements(source);
        assert_eq!(statements.len(), 4);
        assert!(statements[0].contains("http://example.com/a;b"));
        assert!(statements[1].contains("title=[a; b]"));
        assert_eq!(statements[2], "close()");
        assert_eq!(statements[3], "run(\"Smooth\")");
    }

    #[test]
    fn parse_source_preserves_double_slashes_inside_strings() {
        let invocations = parse_macro_source(
            r#"open("https://example.com/a//b.tif"); close();"#,
            &command_registry::command_catalog(),
        )
        .expect("source should parse");
        assert_eq!(invocations.len(), 2);
        assert_eq!(
            invocations[0].params,
            Some(json!({ "url": "https://example.com/a//b.tif" }))
        );
    }

    #[test]
    fn parse_source_ignores_named_macro_block_declarations() {
        let invocations = parse_macro_source(
            r#"
                macro "Quick Smooth" {
                    run("Smooth");
                    close();
                }
            "#,
            &command_registry::command_catalog(),
        )
        .expect("named macro should parse");
        assert_eq!(
            invocations
                .iter()
                .map(|invocation| invocation.command_id.as_str())
                .collect::<Vec<_>>(),
            ["process.smooth", "file.close"]
        );
    }

    #[test]
    fn parse_source_accepts_inline_named_macro_blocks() {
        let invocations = parse_macro_source(
            r#"macro "Quick Smooth" { run("Smooth"); }"#,
            &command_registry::command_catalog(),
        )
        .expect("inline named macro should parse");
        assert_eq!(invocations, [invocation("process.smooth", None)]);
    }

    #[test]
    fn string_literals_preserve_unicode() {
        assert_eq!(
            parse(r#"open("/tmp/Mikroskopie-µm.tif");"#).params,
            Some(json!({ "path": "/tmp/Mikroskopie-µm.tif" }))
        );
    }

    #[test]
    fn recorder_emits_stable_imagej_run_lines() {
        let catalog = command_registry::command_catalog();
        let line = macro_record_line_for_command(
            "process.enhance_contrast",
            Some(&json!({
                "saturated": 0.35,
                "normalize": true,
                "title": "My Image",
                "optional": null,
            })),
            &catalog,
        )
        .expect("recordable command");
        assert_eq!(
            line,
            r#"run("Enhance Contrast...", "normalize saturated=0.35 title=[My Image]");"#
        );
    }

    #[test]
    fn recorder_preserves_plane_stack_scope_for_replay() {
        let catalog = command_registry::command_catalog();
        let params = json!({ "__image_rs_process_stack": true });
        let line = macro_record_line_for_command("process.smooth", Some(&params), &catalog)
            .expect("Smooth should be recordable");

        assert_eq!(line, r#"run("Smooth", "__image_rs_process_stack");"#);
        assert_eq!(parse(&line), invocation("process.smooth", Some(params)));
    }

    #[test]
    fn recorder_emits_native_display_range_calls() {
        let catalog = command_registry::command_catalog();
        let set_range = macro_record_line_for_command(
            "macro.set_min_and_max",
            Some(&json!({ "minimum": -12.5, "maximum": 4095.0 })),
            &catalog,
        )
        .expect("display range should be recordable");
        assert_eq!(set_range, "setMinAndMax(-12.5, 4095);");
        assert_eq!(
            parse(&set_range),
            invocation(
                "macro.set_min_and_max",
                Some(json!({ "minimum": -12.5, "maximum": 4095.0 }))
            )
        );

        let reset_range = macro_record_line_for_command("macro.reset_min_and_max", None, &catalog)
            .expect("display range reset should be recordable");
        assert_eq!(reset_range, "resetMinAndMax();");
        assert_eq!(
            parse(&reset_range),
            invocation("macro.reset_min_and_max", None)
        );
    }

    #[test]
    fn install_macro_copies_only_supported_files() {
        let directory = tempdir().expect("temp directory");
        let source = directory.path().join("Example.txt");
        let install_dir = directory.path().join("installed");
        fs::write(&source, r#"run("Smooth");"#).expect("write source macro");

        let installed =
            install_macro_file_to_dir(&source, &install_dir).expect("install macro file");
        assert_eq!(installed, install_dir.join("Example.ijm"));
        assert_eq!(
            fs::read_to_string(installed).expect("read installed macro"),
            r#"run("Smooth");"#
        );
        assert!(installed_macro_file_name(Path::new("Plugin.jar")).is_err());
    }
}
