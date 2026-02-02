
// vertex shader

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) tex_coords: vec2f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) tex_coords: vec2f
}

@vertex
fn vertex_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = vec4f(model.position, 1.0);
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
}
