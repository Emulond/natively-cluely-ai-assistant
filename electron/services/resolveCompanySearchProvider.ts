// electron/services/resolveCompanySearchProvider.ts
// Single source of truth for the company-research search provider cascade:
//   Tavily (user key) → Natively API proxy (Natively key / trial token) → null (LLM-only).
// Used by both the manual profile:research-company IPC handler and the automatic
// AOT pipeline (injected via KnowledgeOrchestrator.setSearchProviderResolver),
// so the two paths cannot drift. Resolve per invocation — never cache the
// result — because keys can be added, changed, or removed mid-session.

import { TRIAL_SENTINEL_KEY } from '../config/constants';
import { CredentialsManager } from './CredentialsManager';

// premium/ is absent in source-available builds, so this cannot be a static
// import: TypeScript fails to resolve it (TS2307) and the build breaks for
// anyone without the private submodule. Every other premium reference in
// electron/ (featureGate.ts, ipcHandlers.ts) uses a lazy require() for the
// same reason. The provider is opaque here — both callers hand it straight
// back to premium APIs (engine.setSearchProvider,
// setSearchProviderResolver) and never call methods on it in this file.
type SearchProvider = object;

export function resolveCompanySearchProvider(): SearchProvider | null {
  const cm = CredentialsManager.getInstance();

  // Each premium require() is wrapped in try/catch, matching featureGate.ts and
  // ipcHandlers.ts. That is not defensive style for its own sake: esbuild bundles
  // this file, and it only tolerates an unresolvable require() when the call sits
  // inside a try/catch (its optional-dependency heuristic). Bare requires abort
  // the build for anyone without the private premium/ submodule. Falling through
  // to null is the documented last step of the cascade — LLM-only research.
  const tavilyApiKey = cm.getTavilyApiKey();
  if (tavilyApiKey) {
    try {
      const {
        TavilySearchProvider,
      } = require('../../premium/electron/knowledge/TavilySearchProvider');
      return new TavilySearchProvider(tavilyApiKey);
    } catch {
      /* premium module not available — fall through to the next tier */
    }
  }

  const nativelyKey = cm.getNativelyApiKey();
  if (nativelyKey) {
    try {
      const {
        NativelySearchProvider,
      } = require('../../premium/electron/knowledge/NativelySearchProvider');
      // Pass the real trial token when the key is the __trial__ sentinel so the
      // server can authenticate via x-trial-token instead of the invalid key.
      const trialToken = nativelyKey === TRIAL_SENTINEL_KEY ? cm.getTrialToken() : undefined;
      console.log('[CompanySearch] Using Natively API search (no Tavily key configured)');
      return new NativelySearchProvider(nativelyKey, trialToken ?? undefined);
    } catch {
      /* premium module not available — fall through to LLM-only */
    }
  }

  return null;
}
