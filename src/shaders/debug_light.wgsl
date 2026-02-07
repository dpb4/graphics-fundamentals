
// vertex shader

struct Camera {
    view_pos: vec4f,
    view_proj: mat4x4f,
}

struct Light {
    position: vec3f,
    // implicit 4 byte padding here because vec3 is always aligned as vec4
    ambient_color: vec3f,
    // implicit 4 byte padding here because vec3 is always aligned as vec4
    diffuse_color: vec3f,
    // implicit 4 byte padding here because vec3 is always aligned as vec4
    specular_color: vec3f,
    // implicit 4 byte padding here because vec3 is always aligned as vec4
}

@group(0) @binding(0)
var<uniform> camera: Camera;
@group(0) @binding(1)
var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) tex_coords: vec2f,
    @location(2) normal: vec3f
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) color: vec3f
}

@vertex
fn vertex_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let scale = 0.25;
    let light_model_position = model.position * scale + light.position;

    out.clip_position = camera.view_proj * vec4f(light_model_position, 1.0);
    out.color = light.diffuse_color;
    return out;
}


// fragment shader

@group(1) @binding(0)
var texture: texture_2d<f32>;
@group(1) @binding(1)
var texture_sampler: sampler;

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
