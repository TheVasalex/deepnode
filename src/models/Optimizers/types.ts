export type GradientDescentOpts = {
  learnRate: number,
  momentumFactor?: number,
  batchSize?:number,
  targetEpoch?:number,
  targetScore?:number,
}

export const OptimizerNames = {
  gradientDescent: "gd",
  adam: "adam"
} as const;
export type OptimizersEnum = (typeof OptimizerNames)[keyof typeof OptimizerNames];