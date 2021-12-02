<template>
  <div v-if="!initialized">
    Initializing...
  </div>
  <div v-if="initialized && initError">
    {{ initError }}
  </div>
  <div
    class="video-frame"
    v-else
  >
    <video
      ref="videoRef"
      autoplay="true"
      muted
      width="600"
      height="500"
    >
    </video>
    <canvas
      class="overlay"
      ref="canvasRef"
      width="600"
      height="500"
    >
    </canvas>
  </div>
</template>
<script lang="ts">
import { defineComponent } from 'vue'
import * as tf from '@tensorflow/tfjs'
import { Rank, Tensor } from '@tensorflow/tfjs'
import { DetectedObject, ObjectDetection } from '@tensorflow-models/coco-ssd'

interface VideoCanvasData {
  initialized: boolean,
  stream: MediaStream | null,
  initError: Record<string, string> | string | null;
}

export default defineComponent({
  name: 'VideoCanvas',
  data (): VideoCanvasData {
    return {
      initialized: false,
      stream: null,
      initError: null
    } as VideoCanvasData
  },
  computed: {
    videoRef (): HTMLVideoElement | null {
      return this.$refs.videoRef as HTMLVideoElement ?? null
    },

    canvasRef (): HTMLCanvasElement | null {
      return this.$refs.canvasRef as HTMLCanvasElement ?? null
    }
  },
  mounted () {
    tf.setBackend('webgl')

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const camera = new Promise((resolve, reject) => {
        navigator.mediaDevices
          .getUserMedia({
            audio: false,
            video: {
              facingMode: 'user'
            }
          })
          .then(stream => {
            this.stream = stream
            if (this.videoRef) {
              this.videoRef.srcObject = stream
              this.videoRef.onloadedmetadata = () => {
                console.log('Metadata loaded')
                resolve(stream)
              }
            }
          })
          .catch(reason => {
            this.initError = reason
            reject(reason)
          })
      })

      const model = import('@tensorflow-models/coco-ssd').then(m => m.load())

      // const modelPromise = load_model()

      Promise.all([camera, model])
        .then(values => {
          this.detectFrame(values[1])
        })
        .catch(error => {
          this.initError = error
        })
        .finally(() => {
          this.initialized = true
        })
    }
  },
  methods: {
    async detectFrame (model: ObjectDetection) {
      if (this.videoRef) {
        // tf.engine().startScope()
        const predictions = await model.detect(this.videoRef)
        this.renderPredictions(predictions)

        tf.nextFrame().then(() => {
          this.detectFrame(model)
        })
        // tf.engine().endScope()
      }
    },

    processInput (videoRef: HTMLVideoElement | null): Tensor<Rank> | null {
      if (videoRef) {
        const tfimg = tf.browser.fromPixels(videoRef).toInt()
        const expandedimg = tfimg.transpose([0, 1, 2]).expandDims()
        return expandedimg
      } else {
        return null
      }
    },

    renderPredictions (predictions: DetectedObject[]) {
      if (this.canvasRef) {
        const context = this.canvasRef?.getContext('2d')
        if (context) {
          context.clearRect(0, 0, this.canvasRef.width, this.canvasRef.height)

          for (let i = 0; i < predictions.length; i++) {
            context.beginPath()
            context.rect(...predictions[i].bbox)
            context.lineWidth = 1
            context.strokeStyle = 'green'
            context.fillStyle = 'green'
            context.stroke()
            context.fillText(
              predictions[i].score.toFixed(3) + ' ' + predictions[i].class, predictions[i].bbox[0],
              predictions[i].bbox[1] > 10 ? predictions[i].bbox[1] - 5 : 10
            )
          }
        }
      }
    }
  }
})
</script>

<style lang="stylus" scoped>
.video-frame
  position relative
  .overlay
    position absolute
    top calc(50% - 250px)
    left calc(50% - 300px)
    width 600px
    height 500px
</style>
