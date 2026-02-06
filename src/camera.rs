use std::time::Duration;

use cgmath::{Deg, InnerSpace, Matrix4, Point3, Rad, Vector3, Vector4, perspective};
use winit::{event::MouseScrollDelta, keyboard::KeyCode};

// wgpu expects NDC where x and y are in [-1, 1] and z in [0, 1]
// whereas opengl has z in [-1, 1]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::from_cols(
    Vector4::new(1.0, 0.0, 0.0, 0.0),
    Vector4::new(0.0, 1.0, 0.0, 0.0),
    Vector4::new(0.0, 0.0, 0.5, 0.0),
    Vector4::new(0.0, 0.0, 0.5, 1.0),
);

pub struct Projection {
    aspect_ratio: f32,
    fov_vertical: f32,
    z_plane_near: f32,
    z_plane_far: f32,
}

impl Projection {
    pub fn new(width: u32, height: u32, fov: f32, z_plane_near: f32, z_plane_far: f32) -> Self {
        Self {
            aspect_ratio: width as f32 / height as f32,
            fov_vertical: fov / (width as f32 / height as f32),
            z_plane_near,
            z_plane_far,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let fov = self.fov_vertical * self.aspect_ratio;
        self.aspect_ratio = width as f32 / height as f32;
        self.fov_vertical = fov / self.aspect_ratio;
    }

    pub fn perspective_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX
            * perspective(
                Deg(self.fov_vertical),
                self.aspect_ratio,
                self.z_plane_near,
                self.z_plane_far,
            )
    }
}

#[derive(Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
}

impl Camera {
    pub fn view_matrix(&self) -> Matrix4<f32> {
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();

        Matrix4::look_to_rh(
            self.position,
            Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize(),
            Vector3::unit_y(),
        )
    }

    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>>(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }
}

pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    amount_yaw: f32,
    amount_pitch: f32,
    amount_scroll: f32,
    speed: f32,
    sensitivity: f32,
}


impl CameraController {
    const SAFE_FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2 - 0.0001;
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            amount_yaw: 0.0,
            amount_pitch: 0.0,
            amount_scroll: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn handle_key(&mut self, code: KeyCode, is_pressed: bool) {
        let amount = if is_pressed {1.0} else {0.0};

        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.amount_forward = amount;
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.amount_left = amount;
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.amount_backward = amount;
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.amount_right = amount;
            }
            KeyCode::Space => {
                self.amount_up = amount;
            }
            KeyCode::ShiftLeft => {
                self.amount_down = amount;
            }
            _ => {},
        }
    }

    pub fn handle_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.amount_yaw = mouse_dx as f32;
        self.amount_pitch = mouse_dy as f32;
    }

    pub fn handle_scroll(&mut self, delta: &MouseScrollDelta) {
        self.amount_scroll = match delta {
            MouseScrollDelta::LineDelta(_, amount) => {
                amount * 100.0
            },
            MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition {
                y: amount,
                ..
            }) => {
                *amount as f32
            }
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32();

        let (sin_yaw, cos_yaw) = camera.yaw.0.sin_cos();
        // calculate the camera's local forward and right vectors
        let forward = Vector3::new(cos_yaw, 0.0, sin_yaw).normalize();
        let right = Vector3::new(-sin_yaw, 0.0, cos_yaw).normalize();
        
        // move the camera with wasd
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;
    
        let (sin_pitch, cos_pitch) = camera.pitch.0.sin_cos();
        // calculate the vector along the camera's line of sight
        let eye_direction = Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();

        // move the camera in/out with scrolling
        camera.position += eye_direction * self.amount_scroll * self.speed * self.sensitivity * dt;

        // move the camera up and down (absolute)
        camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        // rotate the camera
        camera.yaw += Rad(self.amount_yaw) * self.sensitivity * dt;
        camera.pitch += Rad(self.amount_pitch) * self.sensitivity * dt;

        // mouse amounts are only called on deltas so they need to be reset
        self.amount_scroll = 0.0;
        self.amount_yaw = 0.0;
        self.amount_pitch = 0.0;

        // avoid gimbal lock by constraining pitch
        if camera.pitch < -Rad(Self::SAFE_FRAC_PI_2) {
            camera.pitch = -Rad(Self::SAFE_FRAC_PI_2);
        } else if camera.pitch > Rad(Self::SAFE_FRAC_PI_2) {
            camera.pitch = Rad(Self::SAFE_FRAC_PI_2);
        }
    }
}
