from typing import Any, Dict, List, Optional

class Memory:
    """
    A standalone class for managing historical and contextual information
    to be shared among agents and other components.
    """
    def __init__(self, initial_data: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the Memory class.

        Args:
            initial_data (Optional[List[Dict[str, Any]]]): Preloaded memory entries.
        """
        self._memory = initial_data or []

    def add_entry(self, data: Dict[str, Any]) -> None:
        """
        Add a new entry to memory.

        Args:
            data (Dict[str, Any]): A dictionary representing a memory entry.
        """
        self._memory.append(data)

    def query(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """
        Query the memory for entries where a key matches a specified value.

        Args:
            key (str): The key to search for.
            value (Any): The value to match.

        Returns:
            List[Dict[str, Any]]: A list of matching memory entries.
        """
        return [entry for entry in self._memory if entry.get(key) == value]

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent memory entry.

        Returns:
            Optional[Dict[str, Any]]: The latest memory entry or None if memory is empty.
        """
        return self._memory[-1] if self._memory else None

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all memory entries.

        Returns:
            List[Dict[str, Any]]: A list of all memory entries.
        """
        return self._memory

    def clear_memory(self) -> None:
        """
        Clear all memory entries.
        """
        self._memory.clear()
