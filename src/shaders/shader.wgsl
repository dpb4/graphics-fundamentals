
// vertex shader

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

@group(1) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) tex_coords: vec2f,
    @location(2) normal: vec3f
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) tex_coords: vec2f
}

@vertex
fn vertex_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = camera.view_proj * vec4f(model.position, 1.0);
    out.tex_coords = model.tex_coords;
    return out;
}


// fragment shader

@group(0) @binding(0)
var texture: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(texture, texture_sampler, in.tex_coords);
    // return vec4f(1.0, 1.0, 0.0, 1.0);
}
