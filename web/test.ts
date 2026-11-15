function describe(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail
  // 422 validation errors: [{"loc": [...], "msg": "..."}]
  if (Array.isArray(detail)) {
    return detail.map((d: { msg?: string }) => d.msg ?? '').join('; ') || fallback
  }
  return fallback
}

const input = ''

console.log(describe(input, 'fallback message'))