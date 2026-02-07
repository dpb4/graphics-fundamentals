
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

struct ModelTransformation {
    model_transform_col0: vec4f,
    model_transform_col1: vec4f,
    model_transform_col2: vec4f,
    model_transform_col3: vec4f,
}

@group(2) @binding(0)
var<uniform> model_transformation: ModelTransformation;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) tex_coords: vec2f,
    @location(2) normal: vec3f
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) tex_coords: vec2f,
    @location(1) world_normal: vec3f,
    @location(2) world_position: vec3f,
}

@vertex
fn vertex_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let model_transformation_matrix = mat4x4(
        model_transformation.model_transform_col0,
        model_transformation.model_transform_col1,
        model_transformation.model_transform_col2,
        model_transformation.model_transform_col3
        );

    let world_position_h = model_transformation_matrix * vec4f(model.position, 1.0);

    out.clip_position = camera.view_proj * world_position_h;
    out.tex_coords = model.tex_coords;
    out.world_normal = model.normal;
    out.world_position = world_position_h.xyz;
    return out;
}


// fragment shader

@group(1) @binding(0)
var texture: texture_2d<f32>;
@group(1) @binding(1)
var texture_sampler: sampler;

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4f {
    let light_direction = normalize(light.position - in.world_position);

    let diffuse_strength = max(dot(in.world_normal, light_direction), 0.0);
    let light_diffuse = light.diffuse_color * diffuse_strength;

    let view_direction = normalize(camera.view_pos.xyz - in.world_position);
    let reflect_direction = reflect(-light_direction, in.world_normal);

    let specular_strength = pow(max(dot(view_direction, reflect_direction), 0.0), 32.0);
    let light_specular = light.specular_color * specular_strength;

    let material_diffuse_color = textureSample(texture, texture_sampler, in.tex_coords).xyz;

    // let output_color = (light_specular);
    let output_color = (light.ambient_color + light_diffuse + light_specular) * material_diffuse_color;

    return vec4f(output_color, 1.0);
}
