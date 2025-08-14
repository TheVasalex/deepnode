import { Initializer } from "./types"
import Glorot from "./Glorot"
import He from "./He"
import LeCun from "./LeCun"
import Zeroes from "./Zeroes"

const Initializers = {
    Glorot,
    LeCun,
    He,
    Zeroes
}

export {Initializers, Initializer as Initiliazer}