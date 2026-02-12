
// vertex shader

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
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let model_transformation_matrix = mat4x4(
        model_transformation.model_transform_col0,
        model_transformation.model_transform_col1,
        model_transformation.model_transform_col2,
        model_transformation.model_transform_col3
    );

    // TODO this only works if the model transformation is orthogonal ie no stretching/skewing
    let normal_transformation_matrix = mat3x3f(model_transformation_matrix[0].xyz, model_transformation_matrix[1].xyz, model_transformation_matrix[2].xyz);

    let world_normal = normalize(normal_transformation_matrix * vertex.normal);
    let world_tangent = normalize(normal_transformation_matrix * vertex.tangent);
    let world_bitangent = normalize(normal_transformation_matrix * vertex.bitangent);

    let tangent_transformation_matrix = transpose(mat3x3f(world_tangent, world_bitangent, world_normal));

    let world_position_h = model_transformation_matrix * vec4f(vertex.position, 1.0);

    out.clip_position = camera.view_proj * world_position_h;
    out.tex_coords = vertex.tex_coords;
    out.world_normal   = world_normal;
    out.world_position = world_position_h.xyz;
    out.tangent_position       = tangent_transformation_matrix * world_position_h.xyz;
    out.tangent_view_position  = tangent_transformation_matrix * camera.view_pos.xyz;
    out.tangent_light_position = tangent_transformation_matrix * light.position;
    // out.tangent_position       = world_normal;
    // out.tangent_view_position  = vertex.tangent;
    // out.tangent_light_position = world_bitangent;
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

    var material_diffuse_color: vec3f;

    if material.has_diffuse_texture == 1{
        material_diffuse_color = textureSample(diffuse_texture, diffuse_sampler, in.tex_coords).xyz;
    } else {
        material_diffuse_color = material.diffuse_color;
    }


    var material_normal: vec3f;

    if material.has_normal_texture == 1{
        material_normal = textureSample(normal_texture, normal_sampler, in.tex_coords).xyz;
    } else {
        material_normal = vec3f(0.0, 0.0, 1.0);
    }

    // lighting vectors:
    // let tangent_normal = material_normal.xyz * 2.0 - 1.0; // map from [0, 1] to [-1, 1]
    let tangent_normal = normalize(material_normal.xyz);
    let light_direction = normalize(in.tangent_light_position - in.tangent_position); // vector from point to light (in tangent space)
    let view_direction = normalize(in.tangent_view_position - in.tangent_position); // vector from point to eye (in tangent space)
    let half_direction = normalize(view_direction + light_direction);
    let reflect_direction = reflect(-light_direction, view_direction);


    let diffuse_strength = max(dot(tangent_normal, light_direction), 0.0);
    let light_diffuse = light.diffuse_color * diffuse_strength;

    // let reflect_direction = reflect(-light_direction, tangent_normal);
    // let specular_strength = pow(max(dot(view_direction, reflect_direction), 0.0), 128.0); // just phong
    let specular_strength = pow(max(dot(tangent_normal, half_direction), 0.0), 32.0); // blinn phong
    // let specular_strength = 0.0;
    let light_specular = light.specular_color * specular_strength;

    // let angle = (dot(tangent_normal, half_direction) + 1.0) * 0.5;
    // let angle = dot(tangent_normal, half_direction);
    // let angle = length(in.tangent_view_position);
    // let angle = dot(view_direction, reflect_direction);
    // let angle = (dot(in.view, half_direction) + 1.0) * 0.5;

    // let output_color = (light_specular);

    let output_color = (light.ambient_color + light_diffuse + light_specular) * material_diffuse_color;
    // var output_color = vec3f(angle, 1.0, 1.0 - angle);

    // if length(tangent_normal) < 0.5 {
    //     output_color = vec3f(0.0, 1.0, 0.0);
    // }

    // if angle <= 0.5 {
    //     output_color = vec3f(0.0, 1.0, 0.0);
    // }

    return vec4f(output_color, 1.0);
}
