use pyo3::prelude::*;
use crate::*;

#[pyclass]
struct PyNetwork {
    network: Network,
}

#[pymethods]
impl PyNetwork {
    #[new]
    fn new(body_ids: Vec<u64>) -> Self {
        Self { network: Network::new(body_ids) }
    }

    fn connect(&mut self, from: usize, to: usize, weight: f32) {
        self.network.connect(from, to, weight)
    }

    fn add_connections(&mut self, connections: Vec<(usize, usize, f32)>) {
        for (from, to, weight) in connections {
            self.network.connect(from, to, weight);
        }
    }

    fn set_input(&mut self, neuron: usize, value: f32) {
        self.network.set_input(neuron, value);
    }

    fn step(&mut self) -> Vec<usize> {
        self.network.step()
    }

    fn get_potential(&self, neuron: usize) -> f32 {
        self.network.neurons[neuron].potential
    }
}

#[pymodule]
fn malecns(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNetwork>()?;

    Ok(())
}
