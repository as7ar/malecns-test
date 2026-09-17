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


neurons, _ = fetch_neurons("DNge104")

print("DNge104:")
print(
    neurons[
        ["bodyId", "instance", "type", "pre", "post"]
    ].to_string(index=False)
)


_, connections = fetch_adjacencies(
    "DNge104",
    None,
    omit_rois=True,
)

_, upstream_connections = fetch_adjacencies(
    None,
    "DNge104",
    omit_rois=True,
)

print()
print(f"Upstream connections: {len(upstream_connections)}")

upstream_ids = set()

for _, row in upstream_connections.iterrows():
    upstream_ids.add(int(row["bodyId_pre"]))
    upstream_ids.add(int(row["bodyId_post"]))

upstream_neurons, _ = fetch_neurons(
    list(upstream_ids)
)

print()
print("Upstream neuron types:")

print(
    upstream_neurons[
        ["bodyId", "instance", "type", "pre", "post"]
    ].head(50).to_string(index=False)
)

print()
print(f"Connections: {len(connections)}")


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


print()
print(f"Neuron count: {len(neuron_ids)}")
print(f"Connection count: {len(rust_connections)}")


print()
print("Mapping check:")

for body_id in neuron_ids[:10]:
    index = id_to_index[body_id]

    print(
        f"bodyId={body_id}, "
        f"index={index}, "
        f"reverse={neuron_ids[index]}"
    )


posts = connections["bodyId_post"].value_counts()
post_ids = posts.head(20).index.astype(int).tolist()

post_neurons, _ = fetch_neurons(post_ids)

print()
print("Top downstream neurons:")
print(
    post_neurons[
        ["bodyId", "instance", "type", "pre", "post"]
    ].to_string(index=False)
)


_, downstream_connections = fetch_adjacencies(
    post_ids,
    None,
    omit_rois=True,
)

print()
print(f"Downstream connections: {len(downstream_connections)}")


all_neuron_ids = set(neuron_ids)

for _, row in downstream_connections.iterrows():
    all_neuron_ids.add(int(row["bodyId_pre"]))
    all_neuron_ids.add(int(row["bodyId_post"]))


print(f"2-hop neuron count: {len(all_neuron_ids)}")


all_neurons, _ = fetch_neurons(list(all_neuron_ids))

print()
print(f"Metadata neurons: {len(all_neurons)}")

print()
print("Neuron types:")

type_counts = all_neurons["type"].value_counts()

print(type_counts.head(50).to_string())

type_map = dict(
    zip(
        all_neurons["bodyId"].astype(int),
        all_neurons["type"].fillna("Unknown"),
    )
)

type_connections = {}

for _, row in downstream_connections.iterrows():
    from_id = int(row["bodyId_pre"])
    to_id = int(row["bodyId_post"])

    from_type = type_map.get(from_id, "Unknown")
    to_type = type_map.get(to_id, "Unknown")

    key = (from_type, to_type)

    type_connections[key] = type_connections.get(key, 0) + 1


top_type_connections = sorted(
    type_connections.items(),
    key=lambda x: x[1],
    reverse=True,
)


print()
print("Top type connections:")

for (from_type, to_type), count in top_type_connections[:30]:
    print(
        f"{from_type} -> {to_type}: "
        f"{count}"
    )