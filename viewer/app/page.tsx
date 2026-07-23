import type { Metadata } from "next";
import NeuralObservatory from "./NeuralObservatory";

export const metadata: Metadata = {
  title: "Hermes Neural Observatory",
  description: "Live read-only view of the Hippocampal Replay Engine.",
};

export default function Home() {
  return <NeuralObservatory />;
}
