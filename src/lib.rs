use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
struct Neuron {
    potential: f32,
}

struct Synapse {
    to: usize,
    weight: f32,
}

struct Network {
    neurons: Vec<Neuron>,
    connections: Vec<Vec<Synapse>>,
}

impl Network {
    fn new(neuron_count: usize) -> Self {
        let neurons = (0..neuron_count)
            .map(|_| Neuron { potential: 0.0 })
            .collect();

        let connections = (0..neuron_count)
            .map(|_| Vec::new())
            .collect();

        Self {
            neurons,
            connections,
        }
    }

    fn connect(&mut self, from: usize, to: usize, weight: f32) {
        self.connections[from].push(Synapse {
            to,
            weight,
        });
    }

    fn step(&mut self) {
        let mut next = vec![0.0; self.neurons.len()];

        for (from, connections) in self.connections.iter().enumerate() {
            let potential = self.neurons[from].potential;

            for connection in connections {
                next[connection.to] += potential * connection.weight;
            }
        }

        for (neuron, potential) in self.neurons.iter_mut().zip(next) {
            neuron.potential = potential;
        }
    }
}

#[pyclass]
struct PyNetwork {
    network: Network,
}

#[pymethods]
impl PyNetwork {
    #[new]
    fn new(neuron_count: usize) -> Self {
        Self {
            network: Network::new(neuron_count),
        }
    }

    fn connect(&mut self, from: usize, to: usize, weight: f32) {
        self.network.connect(from, to, weight);
    }

    fn add_connections(
        &mut self,
        connections: Vec<(usize, usize, f32)>,
    ) {
        for (from, to, weight) in connections {
            self.connect(from, to, weight);
        }
    }

    fn set_input(&mut self, neuron: usize, value: f32) {
        self.network.neurons[neuron].potential = value;
    }

    fn step(&mut self) {
        self.network.step();
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
