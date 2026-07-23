# Hermes Neural Observatory for Obsidian

This desktop-only adapter embeds the loopback observatory in an Obsidian tab.
It is intentionally read-only: it has no commands that alter memories, neural
weights, or vault notes. Lifecycle changes continue through the versioned core
service, which projects accepted state back into Obsidian with an undo journal.

Install the three files in:

`.obsidian/plugins/hermes-neural-observatory/`

Then enable **Hermes Neural Observatory** in Obsidian and run **Open Neural
Observatory** from the command palette.
