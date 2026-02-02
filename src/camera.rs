use winit::keyboard::KeyCode;

// wgpu expects NDC where x and y are in [-1, 1] and z in [0, 1]
// whereas opengl has z in [-1, 1]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
);

pub struct Camera {
    pub eye_pos: cgmath::Point3<f32>,
    pub target_pos: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect_ratio: f32,
    pub fov_vertical: f32,
    pub z_plane_near: f32,
    pub z_plane_far: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye_pos, self.target_pos, self.up);
        let proj = cgmath::perspective(
            cgmath::Deg(self.fov_vertical),
            self.aspect_ratio,
            self.z_plane_near,
            self.z_plane_far,
        );

        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }

    pub fn new(aspect_ratio: f32, fov: f32) -> Self {
        Self {
            eye_pos: (0.0, 1.0, 2.0).into(),
            target_pos: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect_ratio,
            fov_vertical: fov / aspect_ratio,
            z_plane_near: 0.1,
            z_plane_far: 100.0,
        }
    }

    pub fn to_uniform(&self) -> CameraUniform {
        CameraUniform {
            view_proj: self.build_view_projection_matrix().into(),
        }
    }
}

pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    pub fn handle_key(&mut self, code: KeyCode, is_pressed: bool) -> bool {
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.is_forward_pressed = is_pressed;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.is_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.is_backward_pressed = is_pressed;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.is_right_pressed = is_pressed;
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.target_pos - camera.eye_pos;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when the camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye_pos += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye_pos -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = camera.target_pos - camera.eye_pos;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target_pos and the eye_pos so
            // that it doesn't change. The eye_pos, therefore, still
            // lies on the circle made by the target_pos and eye_pos.
            camera.eye_pos =
                camera.target_pos - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye_pos =
                camera.target_pos - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}
