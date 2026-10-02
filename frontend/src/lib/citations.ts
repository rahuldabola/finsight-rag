/** "[3]" -> a markdown link we can intercept; "[1][4]" becomes two links. Already-linked "[3](...)" is left alone. */
export const linkCitations = (text: string) => text.replace(/\[(\d{1,2})\](?!\()/g, '[$1](#cite-$1)')

/** The source number behind a `#cite-N` link target, or null for ordinary links. */
export const citationNumber = (href: string | undefined): number | null => {
  const m = href?.match(/^#cite-(\d+)$/)
  return m ? Number(m[1]) : null
}
