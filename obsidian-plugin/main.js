const { ItemView, Notice, Plugin } = require("obsidian");

const VIEW_TYPE = "hermes-neural-observatory";
const OBSERVATORY_URL = "http://localhost:3000";

class NeuralObservatoryView extends ItemView {
  getViewType() {
    return VIEW_TYPE;
  }

  getDisplayText() {
    return "Neural Observatory";
  }

  getIcon() {
    return "brain-circuit";
  }

  async onOpen() {
    const container = this.containerEl.children[1];
    container.empty();
    container.addClass("hermes-observatory-container");
    const frame = container.createEl("iframe", {
      attr: {
        title: "Hermes Neural Observatory",
        src: OBSERVATORY_URL,
        sandbox: "allow-scripts allow-same-origin",
        referrerpolicy: "no-referrer",
      },
    });
    frame.addEventListener("error", () => {
      new Notice("Start the local Neural Observatory service, then reopen this view.");
    });
  }
}

module.exports = class HermesNeuralObservatoryPlugin extends Plugin {
  async onload() {
    this.registerView(VIEW_TYPE, (leaf) => new NeuralObservatoryView(leaf));
    this.addRibbonIcon("brain-circuit", "Open Neural Observatory", () =>
      this.activateView(),
    );
    this.addCommand({
      id: "open-neural-observatory",
      name: "Open Neural Observatory",
      callback: () => this.activateView(),
    });
  }

  async onunload() {
    this.app.workspace.detachLeavesOfType(VIEW_TYPE);
  }

  async activateView() {
    const workspace = this.app.workspace;
    let leaf = workspace.getLeavesOfType(VIEW_TYPE)[0];
    if (!leaf) {
      leaf = workspace.getLeaf("tab");
      await leaf.setViewState({ type: VIEW_TYPE, active: true });
    }
    workspace.revealLeaf(leaf);
  }
};
