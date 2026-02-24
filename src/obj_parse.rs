use std::collections::HashMap;

use crate::model;

#[derive(Debug)]
pub enum OBJLoadError {
    FileNotFound(std::io::Error),
    Parse(String, usize, String),
}

#[derive(Debug)]
pub enum MTLLoadError {
    FileNotFound(std::io::Error),
    Parse(String, usize, String),
    MtlNotFound(String),
}

#[derive(Debug)]
pub struct ParsedOBJ {
    pub model_verts: Vec<model::ModelVertex>,
    pub raw_verts: Vec<(f32, f32, f32)>,
    pub raw_uvs: Vec<(f32, f32)>,
    pub raw_normals: Vec<(f32, f32, f32)>,
    pub indices: Vec<u32>,
    pub material: Option<String>,
    pub material_lib: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct ParsedMTL {
    pub name: Option<String>,
    pub ka: Option<[f32; 3]>,
    pub kd: Option<[f32; 3]>,
    pub ks: Option<[f32; 3]>,
    pub ns: Option<f32>,
    pub d: Option<f32>,
    pub ni: Option<f32>,
    pub illum: Option<u16>,
    pub map_bump: Option<String>,
    pub map_kd: Option<String>,
}

impl std::fmt::Display for OBJLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OBJLoadError::FileNotFound(error) => {
                write!(f, "IO error while loading OBJ file:\n{}", error)
            }
            OBJLoadError::Parse(filepath, line_num, msg) => write!(
                f,
                "Error loading OBJ file {}:\nline {}: {}",
                filepath, line_num, msg
            ),
        }
    }
}

impl std::fmt::Display for ParsedOBJ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "model verts: {}\nraw verts: {}\nraw uvs: {}\nraw normals: {}\nindices: {} ({} triangles)\nmaterial: {}\nmaterial lib: {}\n",
            self.model_verts.len(),
            self.raw_verts.len(),
            self.raw_uvs.len(),
            self.raw_normals.len(),
            self.indices.len(),
            self.indices.len() / 3,
            self.material.as_ref().unwrap_or(&"none".to_string()),
            self.material_lib.as_ref().unwrap_or(&"none".to_string()),
        )
    }
}

fn parse_vector_line(line: &str) -> Result<Vec<f32>, std::num::ParseFloatError> {
    line.split_ascii_whitespace()
        .skip(1)
        .map(|s| s.parse::<f32>())
        .collect()
}

fn parse_face_line(line: &str) -> Result<Vec<Vec<u32>>, std::num::ParseIntError> {
    Ok(line
        .split_ascii_whitespace()
        .skip(1)
        .map(|ft| {
            let mut fv = ft
                .split("/")
                .map(|i| i.parse::<u32>().unwrap_or(1))
                .collect::<Vec<u32>>();
            fv.resize(3, 1);
            fv
        })
        .collect())
}

pub fn parse_obj(filepath: &str) -> Result<ParsedOBJ, OBJLoadError> {
    let file = std::fs::read_to_string(filepath).map_err(|e| OBJLoadError::FileNotFound(e))?;

    let mut raw_verts: Vec<(f32, f32, f32)> = Vec::new();
    let mut raw_uvs: Vec<(f32, f32)> = Vec::new();
    let mut raw_normals: Vec<(f32, f32, f32)> = Vec::new();

    let mut face_vert_index_map = HashMap::new();
    let mut indices = Vec::new();

    let mut model_verts = Vec::new();

    let mut material = None;
    let mut material_lib = None;

    for (linenum, line) in file.lines().enumerate() {
        if line.starts_with("#") {
            continue;
        } else if line.starts_with("f") {
            if let Ok(vvi) = parse_face_line(line) {
                for face_vert in vvi {
                    let key = (face_vert[0], face_vert[1], face_vert[2]);

                    let index = match face_vert_index_map.get(&key) {
                        Some(&i) => i,
                        None => {
                            let i = model_verts.len();
                            model_verts.push(model::ModelVertex {
                                position: raw_verts[key.0 as usize - 1].into(),
                                tex_coords: (*raw_uvs
                                    .get(key.1 as usize - 1)
                                    .unwrap_or(&(0.0, 0.0)))
                                .into(),
                                normal: (*raw_normals
                                    .get(key.2 as usize - 1)
                                    .unwrap_or(&(0.0, 0.0, 0.0)))
                                .into(),
                                tangent: [0.0; 3],
                                bitangent: [0.0; 3],
                            });
                            face_vert_index_map.insert(key, i);
                            i
                        }
                    };
                    indices.push(index as u32);
                }
            } else {
                return Err(OBJLoadError::Parse(
                    filepath.to_string(),
                    linenum,
                    "could not parse faces".to_string(),
                ));
            }
        } else if line.starts_with("v") {
            match parse_vector_line(line) {
                Ok(linevec) => {
                    if line.starts_with("vn") {
                        raw_normals.push((linevec[0], linevec[1], linevec[2]));
                    } else if line.starts_with("vt") {
                        raw_uvs.push((linevec[0], linevec[1]));
                    } else {
                        raw_verts.push((linevec[0], linevec[1], linevec[2]));
                    }
                }
                Err(e) => {
                    return Err(OBJLoadError::Parse(
                        file.to_string(),
                        linenum,
                        "could not parse float: ".to_string() + &e.to_string(),
                    ));
                }
            }
        } else {
            if line.starts_with("mtllib") {
                material_lib = line
                    .split_ascii_whitespace()
                    .skip(1)
                    .next()
                    .map(|s| s.to_string());
            } else if line.starts_with("usemtl") {
                material = line
                    .split_ascii_whitespace()
                    .skip(1)
                    .next()
                    .map(|s| s.to_string());
            }
        }
    }

    Ok(ParsedOBJ {
        model_verts,
        raw_verts,
        raw_uvs,
        raw_normals,
        indices,
        material,
        material_lib,
    })
}

fn parse_float_line(line: &str) -> Result<f32, std::num::ParseFloatError> {
    line.split_ascii_whitespace()
        .nth(1)
        .unwrap_or("")
        .parse::<f32>()
}

fn parse_mtl_line(
    parsed: &mut ParsedMTL,
    line: &str,
    linenum: usize,
    filepath: &str,
) -> Result<(), MTLLoadError> {
    let err_closure = |s| {
        Err::<(), MTLLoadError>(MTLLoadError::Parse(
            filepath.to_string(),
            linenum,
            "could not parse".to_string() + s,
        ))
    };

    if line.starts_with("Ka") {
        match parse_vector_line(line) {
            Ok(v) => {
                parsed.ka = Some([v[0], v[1], v[2]]);
            }
            Err(_) => {
                return err_closure("Ka");
            }
        }
    } else if line.starts_with("Kd") {
        match parse_vector_line(line) {
            Ok(v) => {
                parsed.kd = Some([v[0], v[1], v[2]]);
            }
            Err(_) => {
                return err_closure("Kd");
            }
        }
    } else if line.starts_with("Ks") {
        match parse_vector_line(line) {
            Ok(v) => {
                parsed.ks = Some([v[0], v[1], v[2]]);
            }
            Err(_) => {
                return err_closure("Ks");
            }
        }
    } else if line.starts_with("Ns") {
        match parse_float_line(line) {
            Ok(f) => {
                parsed.ns = Some(f);
            }
            Err(_) => {
                return err_closure("Ns");
            }
        }
    } else if line.starts_with("d") {
        match parse_float_line(line) {
            Ok(f) => {
                parsed.d = Some(f);
            }
            Err(_) => {
                return err_closure("d");
            }
        }
    } else if line.starts_with("ni") {
        match parse_float_line(line) {
            Ok(f) => {
                parsed.ni = Some(f);
            }
            Err(_) => {
                return err_closure("ni");
            }
        }
    } else if line.starts_with("illum") {
        match parse_float_line(line) {
            Ok(f) => {
                parsed.illum = Some(f as u16);
            }
            Err(_) => {
                return err_closure("illum");
            }
        }
    } else if line.starts_with("map_Bump") {
        parsed.map_bump = line
            .split_ascii_whitespace()
            .skip(1)
            .next()
            .map(|s| s.to_string());
    } else if line.starts_with("map_Kd") {
        parsed.map_kd = line
            .split_ascii_whitespace()
            .skip(1)
            .next()
            .map(|s| s.to_string());
    }

    Ok(())
}

pub fn parse_mtl(filepath: &str, name: &str) -> Result<ParsedMTL, MTLLoadError> {
    let file = std::fs::read_to_string(filepath).map_err(|e| MTLLoadError::FileNotFound(e))?;

    let mut parsed = ParsedMTL::default();

    let mtl_line_match = format!("newmtl {}", name);
    let mut match_found = false;

    for (linenum, line) in file.lines().enumerate() {
        if line.starts_with("#") {
            continue;
        } else if line.starts_with("newmtl") {
            if line.starts_with(&mtl_line_match) {
                parsed.name = Some(name.to_string());
                match_found = true;
            } else {
                if match_found {
                    break;
                }
            }
        } else if match_found {
            parse_mtl_line(&mut parsed, line, linenum, filepath)?;
        }
    }

    if !match_found {
        Err(MTLLoadError::MtlNotFound(format!(
            "{} not found in {}",
            name, filepath
        )))
    } else {
        Ok(parsed)
    }
}

pub fn parse_all_mtls(filepath: &str) -> Result<Vec<ParsedMTL>, MTLLoadError> {
    let file = std::fs::read_to_string(filepath).map_err(|e| MTLLoadError::FileNotFound(e))?;

    let mut all_parsed = Vec::new();
    let mut current_parsed = ParsedMTL::default();

    let mut first_mtl = true;

    for (linenum, line) in file.lines().enumerate() {
        if line.starts_with("#") {
            continue;
        } else if line.starts_with("newmtl") {
            if first_mtl {
                first_mtl = false;
            } else {
                all_parsed.push(current_parsed.clone());
                current_parsed = ParsedMTL::default();
            }
            current_parsed.name = line.split_ascii_whitespace().nth(1).map(|s| s.to_string());
        } else {
            parse_mtl_line(&mut current_parsed, line, linenum, filepath)?;
        }
    }

    all_parsed.push(current_parsed);

    Ok(all_parsed)
}
