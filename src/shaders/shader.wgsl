
// vertex shader

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) color: vec3f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) color: vec3f
}

@vertex
fn vertex_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = vec4f(model.position, 1.0);
    out.color = model.color;
    return out;
}


// fragment shader

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
