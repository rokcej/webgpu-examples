struct VertexOut { 
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f
}

@group(0) @binding(0) var uTexture: texture_2d<f32>;
@group(0) @binding(1) var uSampler: sampler;

const vertices = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0)
);

@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
    var vertex: vec2f = vertices[vertexIndex];

    var vertexOut: VertexOut;
    vertexOut.position = vec4f(vertex, 0.0, 1.0);
    vertexOut.uv = vertex * vec2f(0.5, -0.5) + 0.5;
    return vertexOut;
}

@fragment
fn fragment_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSample(uTexture, uSampler, uv);
}
