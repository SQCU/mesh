class RowWindow:
    def __init__(self, capacity, mass):
        self.capacity = max(1, int(capacity))
        self.mass = mass
        self.entries = {}
        self.row_mass = 0
        self.ingested_row_mass = 0
        self.evicted_row_mass = 0

    def clear(self):
        self.entries.clear()
        self.row_mass = 0
        self.ingested_row_mass = 0
        self.evicted_row_mass = 0

    def put(self, key, value, retain=()):
        value_mass = int(self.mass(value))
        self.capacity = max(self.capacity, value_mass)
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.row_mass -= int(self.mass(previous))
        self.entries[key] = value
        self.row_mass += value_mass
        self.ingested_row_mass += value_mass
        retained = set(retain)
        retained.add(key)
        while self.entries and self.row_mass > self.capacity:
            oldest = next((candidate for candidate in self.entries if candidate not in retained), None)
            if oldest is None:
                self.capacity = self.row_mass
                break
            removed = self.entries.pop(oldest)
            removed_mass = int(self.mass(removed))
            self.row_mass -= removed_mass
            self.evicted_row_mass += removed_mass

    def get(self, key, default=None):
        return self.entries.get(key, default)

    def items(self):
        return self.entries.items()

    def values(self):
        return self.entries.values()

    def measure(self):
        return {
            "row_capacity": self.capacity,
            "ingested_row_mass": self.ingested_row_mass,
            "retained_row_mass": self.row_mass,
            "evicted_row_mass": self.evicted_row_mass,
            "sequence_mass": len(self.entries),
        }

    def __bool__(self):
        return bool(self.entries)
