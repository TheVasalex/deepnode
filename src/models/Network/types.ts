import { LossFunctionsEnum } from "../Functions";
import { LayerData } from "../Layer/types";
import Network from "./Network";

export type NetworkData = {
  layers: LayerData[];
  lossFunctionName: LossFunctionsEnum;
};

export default Network;
