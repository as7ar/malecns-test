import os

from dotenv import load_dotenv
from neuprint import Client, fetch_neurons, fetch_adjacencies

import malecns

load_dotenv()

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=os.getenv("TOKEN"),
)

neurons, roi_counts = fetch_neurons("DNge104")
neurons, connections = fetch_adjacencies(
    "DNge104",
    None,
    omit_rois=True,
)

neuron_ids = set()

for _, row in connections.iterrows():
    neuron_ids.add(int(row["bodyId_pre"]))
    neuron_ids.add(int(row["bodyId_post"]))

print("Neuron count:", len(neuron_ids))

neuron_ids = list(neuron_ids)

id_to_index = {
    neuron_id: index
    for index, neuron_id in enumerate(neuron_ids)
}

row = connections.iloc[0]

from_id = int(row["bodyId_pre"])
to_id = int(row["bodyId_post"])
weight = float(row["weight"]) / 100.0

from_index = id_to_index[from_id]
to_index = id_to_index[to_id]

rust_connections = []

for _, row in connections.iterrows():
    from_id = int(row["bodyId_pre"])
    to_id = int(row["bodyId_post"])
    weight = float(row["weight"]) / 100.0

    from_index = id_to_index[from_id]
    to_index = id_to_index[to_id]

    rust_connections.append(
        (from_index, to_index, weight)
    )

print("Connections(", len(rust_connections),")")

network = malecns.PyNetwork(len(neuron_ids))

network.add_connections(rust_connections)

print("neurons:", len(neuron_ids))
print("connections:", len(rust_connections))

network.set_input(0, 1.0)

for step in range(10):
    spikes = network.step()

    print(
        f"step={step}, "
        f"spike_count={len(spikes)}, "
        f"spikes={spikes[:10]}"
    )