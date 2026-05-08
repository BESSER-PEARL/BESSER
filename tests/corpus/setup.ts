/**
 * jsdom + canvas mocks copied from the @besser/wme library setup, since
 * the library's bundled `dist/index.js` touches `document` and canvas
 * APIs at import time.
 */

class MockCanvasRenderingContext2D {
  font = ""
  measureText(text: string) {
    return { width: text.length * 8 }
  }
  fillRect() {}
  clearRect() {}
  getImageData() {
    return { data: [] }
  }
  putImageData() {}
  createImageData() {
    return { data: [] }
  }
  setTransform() {}
  resetTransform() {}
  drawImage() {}
  save() {}
  restore() {}
  beginPath() {}
  moveTo() {}
  lineTo() {}
  closePath() {}
  stroke() {}
  fill() {}
  translate() {}
  scale() {}
  rotate() {}
  arc() {}
  arcTo() {}
  rect() {}
  clip() {}
}

HTMLCanvasElement.prototype.getContext = function () {
  return new MockCanvasRenderingContext2D() as unknown as CanvasRenderingContext2D
} as unknown as typeof HTMLCanvasElement.prototype.getContext

class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
;(global as any).ResizeObserver = MockResizeObserver

if (typeof PointerEvent === "undefined") {
  class MockPointerEvent extends MouseEvent {
    pointerId: number
    pointerType: string
    width: number
    height: number
    pressure: number
    isPrimary: boolean

    constructor(type: string, params: PointerEventInit = {}) {
      super(type, params)
      this.pointerId = params.pointerId ?? 0
      this.pointerType = params.pointerType ?? ""
      this.width = params.width ?? 1
      this.height = params.height ?? 1
      this.pressure = params.pressure ?? 0
      this.isPrimary = params.isPrimary ?? false
    }
  }
  ;(global as any).PointerEvent = MockPointerEvent
}

Object.defineProperty(SVGElement.prototype, "getBBox", {
  value: () => ({ x: 0, y: 0, width: 100, height: 20 }),
  writable: true,
})

Object.defineProperty(SVGElement.prototype, "getComputedTextLength", {
  value: () => 100,
  writable: true,
})
