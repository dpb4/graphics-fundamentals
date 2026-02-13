
// vertex shader

// TODO add to shader:
// - timestep or framecounter
// - performance tracking
// 

struct Camera {
    view_pos: vec4f,
    view_proj: mat4x4f,
}

struct Light {
    position: vec3f,
    ambient_color: vec3f,
    diffuse_color: vec3f,
    specular_color: vec3f,
}

@group(0) @binding(0)
var<uniform> camera: Camera;
@group(0) @binding(1)
var<uniform> light: Light;

struct ModelTransformation {
    model_transform_col0: vec4f,
    model_transform_col1: vec4f,
    model_transform_col2: vec4f,
    model_transform_col3: vec4f,
}

@group(2) @binding(0)
var<uniform> model_transformation: ModelTransformation;

struct DebugVector {
    position: vec4f,
    direction: vec4f,
}

@group(2) @binding(1)
var<storage, read> debug_vectors: array<DebugVector>;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) tex_coords: vec2f,
    @location(2) normal: vec3f,
    @location(3) tangent: vec3f,
    @location(4) bitangent: vec3f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) tex_coords: vec2f,
    @location(1) world_normal: vec3f,
    @location(2) world_position: vec3f,
    @location(3) tangent_position: vec3f,
    @location(4) tangent_light_position: vec3f,
    @location(5) tangent_view_position: vec3f,
}

@vertex
fn vertex_main(vertex: VertexInput, @builtin(instance_index) i: u32) -> VertexOutput {
    var out: VertexOutput;

    let model_transformation_matrix = mat4x4(
        model_transformation.model_transform_col0,
        model_transformation.model_transform_col1,
        model_transformation.model_transform_col2,
        model_transformation.model_transform_col3
    );

    let dv = debug_vectors[i];

    let dir = normalize(dv.direction.xyz);

    let I = mat3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let a = vec3f(0.0, 1.0, 0.0);
    let b = dir;
    let v = cross(a, b);
    let vx = mat3x3(0.0, v.z, -v.y, -v.z, 0, v.x, v.y, -v.x, 0.0);

    let rot = I + vx + ((1.0 / (1 + dot(a, b))) * vx * vx);

    let position_offset = (rot * vertex.position * 0.0004) + dv.position.xyz;

    let world_position_h = model_transformation_matrix * vec4f(position_offset, 1.0);

    out.clip_position = camera.view_proj * world_position_h;

    return out;
}


// fragment shader

struct Material {
    @size(16) ambient_color: vec3f,
    @size(16) diffuse_color: vec3f,
    @size(16) specular_color: vec3f,

    has_diffuse_texture: u32,
    has_normal_texture: u32,

    @size(8) _tail_pad: u32,
}

@group(1) @binding(0)
var diffuse_texture: texture_2d<f32>;
@group(1) @binding(1)
var diffuse_sampler: sampler;
@group(1) @binding(2)
var normal_texture: texture_2d<f32>;
@group(1) @binding(3)
var normal_sampler: sampler;
@group(1) @binding(4)
var<uniform> material: Material;

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(material.diffuse_color, 1.0);
}
