import { Vector } from "../models/Matrix";
import fs from "node:fs"

export function batchSplit(input: any[], batchSize: number) {
    const batches = [];
    for (let i = 0; i < input.length; i += batchSize) {
      const batch = input.slice(i, i + batchSize);
      batches.push(batch);
    }
    return batches;
  }

  export function getRandomInteger(min: number, max: number): number {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  export function getRandomFloat(min: number, max: number): number {
    min = Math.ceil(min);
    max = Math.floor(max);
    return parseFloat((Math.random() * (max - min + 1) + min).toFixed(2))
  }

  export function getTimeString() {
    let d = new Date();
    let h = (d.getHours()<10?'0':'') + d.getHours();
    let m = (d.getMinutes()<10?'0':'') + d.getMinutes();
    let s = (d.getSeconds()<10?'0':'') + d.getSeconds();
    let timeString = h + ':' + m + ":" + s;
    return timeString;
}

export function parseCSV(csvPath: string, inputCollumnNames: string[], labelNames: string[], csvSeparator: string = ';') {
  const allFileContents = fs.readFileSync(csvPath, "utf-8");
  const lines = allFileContents.split(/\r?\n/);
  const headers = lines[0].split(csvSeparator);
  const inputs = new Array();
  const labels = new Array();

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(csvSeparator);
    const input = new Vector(inputCollumnNames.length);
    const label = new Vector(labelNames.length);
    let lastInputIndex = 0;
    let lastLabelIndex = 0;

    for (let v = 0; v < values.length; v++) {
      const collumnName = headers[v];
      if (inputCollumnNames.includes(collumnName)) {
        input.set(lastInputIndex, 0, Number(values[v]));
        lastInputIndex++;
      } else if (labelNames.includes(collumnName)) {
        label.set(lastLabelIndex, 0, Number(values[v]));
        lastLabelIndex++;
      } else {
        throw new Error(`Could not find collumn: ${collumnName}`);
      }
    }
    inputs.push(input);
    labels.push(label);
  }
  return [inputs, labels];
}