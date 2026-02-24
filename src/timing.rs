use std::collections::VecDeque;

pub struct RollingAverage {
    samples: VecDeque<f32>,
    window_size: usize,
    running_avg: f32,
}

impl RollingAverage {
    pub fn new(window_size: usize) -> Self {
        Self {
            samples: VecDeque::from(vec![0.0; window_size]),
            window_size,
            running_avg: 0.0
        }
    }
    
    pub fn push(&mut self, val: f32) {
        self.samples.push_back(val / self.window_size as f32);
        self.running_avg += val / self.window_size as f32;
        if self.samples.len() > self.window_size as usize {
            self.running_avg -= self.samples.pop_front().unwrap();
        }
    }

    pub fn get(&self) -> f32 {
        self.running_avg
    }
}