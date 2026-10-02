/** Tailwind classes for a company badge; unknown companies get a neutral one. */
const COMPANY_COLORS: Record<string, string> = {
  Infosys: 'bg-sky-400/15 text-sky-300',
  Wipro: 'bg-violet-400/15 text-violet-300',
  Cognizant: 'bg-amber-400/15 text-amber-300',
  Accenture: 'bg-pink-400/15 text-pink-300',
}

export const companyClass = (c: string) => COMPANY_COLORS[c] ?? 'bg-ink-600/60 text-ink-100'
