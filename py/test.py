import malecns

network = malecns.PyNetwork(2)

network.connect(0, 1, 0.5)

for step in range(50):
    network.set_input(0, 0.11)

    spikes = network.step()

    print(
        f"step={step}, "
        f"spikes={spikes}, "
        f"neuron0={network.get_potential(0)}, "
        f"neuron1={network.get_potential(1)}"
    )