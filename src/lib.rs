struct Neuron {
    potential: f32,
    threshold: f32,
    leak: f32,
}

impl Neuron {
    fn new() -> Self {
        Self {
            potential: 0.0,
            threshold: 1.0,
            leak: 0.9,
        }
    }

    fn step(&mut self, input: f32) -> bool {
        self.potential = self.potential * self.leak + input;

        if self.potential < self.threshold {
            return false;
        }

        self.potential = 0.0;

        true
    }
}

struct Synapse {
    to: usize,
    weight: f32,
}

struct Network {
    neurons: Vec<Neuron>,
    connections: Vec<Vec<Synapse>>,
    inputs: Vec<f32>,
}

impl Network {
    fn new(count: usize) -> Self {
        Self {
            neurons: (0..count).map(|_| Neuron::new()).collect(),
            connections: (0..count).map(|_| Vec::new()).collect(),
            inputs: vec![0.0; count],
        }
    }

    fn connect(&mut self, from: usize, to: usize, weight: f32) {
        self.connections[from].push(Synapse { to, weight });
    }

    fn set_input(&mut self, neuron: usize, value: f32) {
        self.inputs[neuron] += value;
    }

    fn step(&mut self) -> Vec<usize> {
        let mut spikes = Vec::new();

        for i in 0..self.neurons.len() {
            let input = self.inputs[i];
            self.inputs[i]=0.0;

            if self.neurons[i].step(input) {
                spikes.push(i);
            }
        }

        for &from in &spikes {
            for synapse in &self.connections[from] {
                self.inputs[synapse.to] += synapse.weight;
            }
        }

        spikes
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn main() {
        let mut network = Network::new(2);

    network.connect(0, 1, 0.6);
    network.set_input(0, 1.0);

    for i in 0..5 {
        let spikes = network.step();

        println!(
            "step={}, spikes={:?}, neuron0={}, neuron1={}",
            i,
            spikes,
            network.neurons[0].potential,
            network.neurons[1].potential,
        );
    }
    }
}