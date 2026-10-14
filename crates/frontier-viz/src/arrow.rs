// SPDX-License-Identifier: Apache-2.0
//! Decode the server's `surface?lod=` response (Arrow IPC stream) into a [`SurfaceMesh`].
//!
//! Accepted layouts (any mix, across one or more record batches; null entries are
//! skipped so sections of different length can share one schema):
//! - `points`: FixedSizeList<Float32|Float64, 3>  or columns `x`, `y`, `z`
//! - `faces`:  FixedSizeList<Int32|UInt32|Int64, 3> or columns `i0`, `i1`, `i2`
//! - any other Float32/Float64 column (optionally prefixed `fields/` or `field_`) is a
//!   scalar field; FixedSizeList<Float32, C> fields are reduced to their magnitude.
//!
//! Both the IPC *stream* and *file* framings are accepted. [`encode`] produces the
//! canonical layout (one schema, batch A = points + point fields, batch B = faces +
//! cell fields) that the mock data source serves.

use crate::VizError;
use crate::surface::SurfaceMesh;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, UInt32Array,
};
use arrow_ipc::reader::{FileReader, StreamReader};
use arrow_schema::DataType;
use std::io::Cursor;

pub fn decode(bytes: &[u8]) -> Result<SurfaceMesh, VizError> {
    let batches = read_batches(bytes)?;
    let mut mesh = SurfaceMesh::default();
    let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut i0, mut i1, mut i2) = (Vec::new(), Vec::new(), Vec::new());
    for batch in &batches {
        let schema = batch.schema();
        for (col, field) in batch.columns().iter().zip(schema.fields()) {
            let name = field.name().as_str();
            let col = &flatten(col.clone());
            match name {
                "points" | "positions" | "xyz" => mesh.points.extend(fixed3(col, floats)?),
                "faces" | "triangles" | "indices" => mesh.faces.extend(fixed3(col, ints)?),
                "x" => xs.extend(floats(col)?),
                "y" => ys.extend(floats(col)?),
                "z" => zs.extend(floats(col)?),
                "i0" | "v0" => i0.extend(ints(col)?),
                "i1" | "v1" => i1.extend(ints(col)?),
                "i2" | "v2" => i2.extend(ints(col)?),
                _ => {
                    let fname = name
                        .strip_prefix("fields/")
                        .or_else(|| name.strip_prefix("field:"))
                        .or_else(|| name.strip_prefix("field_"))
                        .unwrap_or(name)
                        .to_string();
                    if let Some(values) = field_values(col) {
                        mesh.fields.entry(fname).or_default().extend(values);
                    }
                }
            }
        }
    }
    if mesh.points.is_empty() && !xs.is_empty() {
        if xs.len() != ys.len() || xs.len() != zs.len() {
            return Err(VizError::Decode("x/y/z column lengths differ".into()));
        }
        mesh.points = xs
            .iter()
            .zip(&ys)
            .zip(&zs)
            .map(|((x, y), z)| [*x, *y, *z])
            .collect();
    }
    if mesh.faces.is_empty() && !i0.is_empty() {
        if i0.len() != i1.len() || i0.len() != i2.len() {
            return Err(VizError::Decode("i0/i1/i2 column lengths differ".into()));
        }
        mesh.faces = i0
            .iter()
            .zip(&i1)
            .zip(&i2)
            .map(|((a, b), c)| [*a, *b, *c])
            .collect();
    }
    if mesh.points.is_empty() || mesh.faces.is_empty() {
        return Err(VizError::Decode("no points/faces columns found".into()));
    }
    let n = mesh.points.len() as u32;
    if mesh.faces.iter().any(|f| f.iter().any(|i| *i >= n)) {
        return Err(VizError::Decode("face index out of range".into()));
    }
    Ok(mesh)
}

/// Unwrap `List<...>` / `LargeList<...>` wrappers (the server stores one row whose list
/// holds the whole array) down to the innermost non-list array.
fn flatten(col: ArrayRef) -> ArrayRef {
    use arrow_array::{LargeListArray, ListArray};
    let mut cur = col;
    loop {
        if let Some(l) = cur.as_any().downcast_ref::<ListArray>() {
            cur = l.values().clone();
        } else if let Some(l) = cur.as_any().downcast_ref::<LargeListArray>() {
            cur = l.values().clone();
        } else {
            return cur;
        }
    }
}

fn read_batches(bytes: &[u8]) -> Result<Vec<RecordBatch>, VizError> {
    let map = |e: arrow_schema::ArrowError| VizError::Decode(e.to_string());
    if bytes.starts_with(b"ARROW1") {
        let reader = FileReader::try_new(Cursor::new(bytes), None).map_err(map)?;
        return reader.collect::<Result<Vec<_>, _>>().map_err(map);
    }
    let reader = StreamReader::try_new(Cursor::new(bytes), None).map_err(map)?;
    reader.collect::<Result<Vec<_>, _>>().map_err(map)
}

/// Non-null float values of a column.
fn floats(col: &ArrayRef) -> Result<Vec<f32>, VizError> {
    if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        return Ok(a.iter().flatten().collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        return Ok(a.iter().flatten().map(|v| v as f32).collect());
    }
    Err(VizError::Decode(format!(
        "expected float column, got {:?}",
        col.data_type()
    )))
}

fn ints(col: &ArrayRef) -> Result<Vec<u32>, VizError> {
    if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        return Ok(a.iter().flatten().map(|v| v as u32).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<UInt32Array>() {
        return Ok(a.iter().flatten().collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        return Ok(a.iter().flatten().map(|v| v as u32).collect());
    }
    Err(VizError::Decode(format!(
        "expected integer column, got {:?}",
        col.data_type()
    )))
}

/// Non-null rows of a FixedSizeList<_, 3> column.
fn fixed3<T: Copy>(
    col: &ArrayRef,
    conv: fn(&ArrayRef) -> Result<Vec<T>, VizError>,
) -> Result<Vec<[T; 3]>, VizError> {
    let list = col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| {
            VizError::Decode(format!(
                "expected FixedSizeList<_, 3>, got {:?}",
                col.data_type()
            ))
        })?;
    if list.value_length() != 3 {
        return Err(VizError::Decode(format!(
            "expected list width 3, got {}",
            list.value_length()
        )));
    }
    let mut out = Vec::with_capacity(list.len());
    for i in 0..list.len() {
        if !list.is_valid(i) {
            continue;
        }
        let row = conv(&list.value(i))?;
        if row.len() == 3 {
            out.push([row[0], row[1], row[2]]);
        }
    }
    Ok(out)
}

fn field_values(col: &ArrayRef) -> Option<Vec<f32>> {
    match col.data_type() {
        DataType::Float32 | DataType::Float64 => floats(col).ok(),
        DataType::FixedSizeList(_, _) => {
            let list = col.as_any().downcast_ref::<FixedSizeListArray>()?;
            let mut out = Vec::with_capacity(list.len());
            for i in 0..list.len() {
                if list.is_valid(i) {
                    let row = floats(&list.value(i)).ok()?;
                    out.push(row.iter().map(|v| v * v).sum::<f32>().sqrt());
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// Encode a mesh in the canonical layout.
pub fn encode(mesh: &SurfaceMesh) -> Result<Vec<u8>, VizError> {
    use arrow_array::types::{Float32Type, Int32Type};
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;
    let map = |e: arrow_schema::ArrowError| VizError::Decode(e.to_string());

    let item_f32 = Arc::new(Field::new("item", DataType::Float32, true));
    let item_i32 = Arc::new(Field::new("item", DataType::Int32, true));
    let mut fields = vec![
        Field::new("points", DataType::FixedSizeList(item_f32, 3), true),
        Field::new("faces", DataType::FixedSizeList(item_i32, 3), true),
    ];
    for name in mesh.fields.keys() {
        fields.push(Field::new(
            format!("fields/{name}"),
            DataType::Float32,
            true,
        ));
    }
    let schema = Arc::new(Schema::new(fields));
    let n_pts = mesh.points.len();
    let n_faces = mesh.faces.len();

    let null_rows_f32 = |n: usize| {
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            (0..n).map(|_| None::<Vec<Option<f32>>>),
            3,
        )
    };
    let null_rows_i32 = |n: usize| {
        FixedSizeListArray::from_iter_primitive::<Int32Type, _, _>(
            (0..n).map(|_| None::<Vec<Option<i32>>>),
            3,
        )
    };
    let null_f32 = |n: usize| Float32Array::from(vec![None::<f32>; n]);

    let points = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        mesh.points.iter().map(|p| Some(p.iter().map(|v| Some(*v)))),
        3,
    );
    let faces = FixedSizeListArray::from_iter_primitive::<Int32Type, _, _>(
        mesh.faces
            .iter()
            .map(|f| Some(f.iter().map(|v| Some(*v as i32)))),
        3,
    );

    let mut a: Vec<ArrayRef> = vec![Arc::new(points), Arc::new(null_rows_i32(n_pts))];
    let mut b: Vec<ArrayRef> = vec![Arc::new(null_rows_f32(n_faces)), Arc::new(faces)];
    for f in mesh.fields.values() {
        if f.len() == n_pts {
            a.push(Arc::new(Float32Array::from(f.clone())));
            b.push(Arc::new(null_f32(n_faces)));
        } else {
            a.push(Arc::new(null_f32(n_pts)));
            b.push(Arc::new(Float32Array::from(f.clone())));
        }
    }

    let mut out = Vec::new();
    {
        let mut writer =
            arrow_ipc::writer::StreamWriter::try_new(&mut out, &schema).map_err(map)?;
        writer
            .write(&RecordBatch::try_new(schema.clone(), a).map_err(map)?)
            .map_err(map)?;
        writer
            .write(&RecordBatch::try_new(schema.clone(), b).map_err(map)?)
            .map_err(map)?;
        writer.finish().map_err(map)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_roundtrip_preserves_mesh_and_fields() {
        let mut mesh = crate::fixtures::ahmed_body(0);
        mesh.fields
            .insert("cell_area".into(), vec![1.0; mesh.faces.len()]);
        let bytes = encode(&mesh).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.points, mesh.points);
        assert_eq!(back.faces, mesh.faces);
        assert_eq!(back.fields["p"], mesh.fields["p"]);
        assert_eq!(back.fields["cell_area"].len(), mesh.faces.len());
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(decode(b"not arrow").is_err());
    }
}

#[cfg(test)]
mod live_tests {
    /// Decode a surface the real server produced: `FRONTIER_SURFACE_FILE=/path cargo test`.
    #[test]
    fn decodes_server_surface_file() {
        let Ok(path) = std::env::var("FRONTIER_SURFACE_FILE") else {
            return;
        };
        let bytes = std::fs::read(path).expect("read surface file");
        if let Ok(reader) =
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(&bytes), None)
        {
            eprintln!("schema: {:#?}", reader.schema());
        }
        let mesh = super::decode(&bytes).expect("decode");
        eprintln!(
            "points={} faces={} fields={:?}",
            mesh.points.len(),
            mesh.faces.len(),
            mesh.fields
                .iter()
                .map(|(k, v)| (k.clone(), v.len()))
                .collect::<Vec<_>>()
        );
        assert!(!mesh.faces.is_empty());
    }
}
