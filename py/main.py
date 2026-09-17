import os

from dotenv import load_dotenv
from neuprint import Client, fetch_adjacencies

import malecns


load_dotenv()

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=os.getenv("TOKEN"),
)


_, connections = fetch_adjacencies(
    "DNge104",
    None,
    omit_rois=True,
)


neuron_ids = set()

for _, row in connections.iterrows():
    neuron_ids.add(int(row["bodyId_pre"]))
    neuron_ids.add(int(row["bodyId_post"]))

neuron_ids = list(neuron_ids)

id_to_index = {
    body_id: index
    for index, body_id in enumerate(neuron_ids)
}


rust_connections = []

for _, row in connections.iterrows():
    from_id = int(row["bodyId_pre"])
    to_id = int(row["bodyId_post"])
    weight = float(row["weight"]) / 100.0

    rust_connections.append(
        (
            id_to_index[from_id],
            id_to_index[to_id],
            weight,
        )
    )


network = malecns.PyNetwork(neuron_ids)
network.add_connections(rust_connections)


print(f"Neurons: {len(neuron_ids)}")
print(f"Connections: {len(rust_connections)}")


dnge104_index = id_to_index[12781]

print()
print("SNN simulation:")

for step in range(100):
    network.set_input(dnge104_index, 0.11)

    spikes = network.step()

    if not spikes:
        continue

    spike_body_ids = [
        neuron_ids[index]
        for index in spikes
    ]

    print(
        f"step={step}, "
        f"spikes={spike_body_ids}"
    )