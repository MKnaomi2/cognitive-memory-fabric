import type { Metadata } from "next";
import NeuralObservatory from "./NeuralObservatory";

export const metadata: Metadata = {
  title: "Hermes Neural Observatory",
  description: "Live read-only view of the hippocampal memory circuit.",
};

export default function Home() {
  return <NeuralObservatory />;
}
