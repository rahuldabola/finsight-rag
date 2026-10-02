/**
 * Incremental Server-Sent Events parser. Feed it text as it arrives; it returns the `data:` payloads of every
 * complete frame (frames end in a blank line) and the unfinished remainder to prepend to the next chunk.
 */
export function parseSseFrames<T>(buffer: string): { events: T[]; rest: string } {
  const events: T[] = []
  let rest = buffer.replace(/\r\n/g, '\n')
  let sep: number
  while ((sep = rest.indexOf('\n\n')) !== -1) {
    const frame = rest.slice(0, sep)
    rest = rest.slice(sep + 2)
    for (const line of frame.split('\n')) {
      if (line.startsWith('data: ')) events.push(JSON.parse(line.slice(6)) as T)
    }
  }
  return { events, rest }
}
