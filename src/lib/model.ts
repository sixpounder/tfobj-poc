import { GraphModel, loadGraphModel } from '@tensorflow/tfjs-converter'

export const loadModel = async (path: string): Promise<GraphModel> => {
  return await loadGraphModel(path)
}

// export const getCOCO = (): Promise<ObjectDetection> => {
//   return import('@tensorflow-models/coco-ssd').then((m) => {
//     return m.load()
//   })
// }
