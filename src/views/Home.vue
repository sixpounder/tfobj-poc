<template>
  <div class="home">
    <VideoCanvas />
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import VideoCanvas from '@/components/VideoCanvas.vue' // @ is an alias to /src

export default defineComponent({
  name: 'Home',
  components: {
    VideoCanvas
  }
})
</script>
