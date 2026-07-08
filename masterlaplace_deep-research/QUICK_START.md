# Installation Rapide — Deep Research Tool

## 🎯 Installation en 60 Secondes

### Pour les Utilisateurs (VS Code installé)

```bash
# 1. Télécharger le VSIX
# Depuis: copilot-deep-research-tool-0.0.1.vsix

# 2. Installer l'extension
code --install-extension ./copilot-deep-research-tool-0.0.1.vsix

# 3. Recharger VS Code
# Ctrl+Shift+P → "Developer: Reload Window"

# 4. Tester
# Ouvrir Copilot Chat (Ctrl+L)
# Taper: #deep_research Find Vulkan best practices
# Réponse: Job créé, rapport généré dans .github/research_reports/
```

### Pour les Développeurs (Tester localement)

```bash
# 1. Cloner/ouvrir le workspace
cd copilot-deep-research-tool

# 2. Compiler
npm run compile

# 3. Exécuter en mode debug
# Appuyer sur F5 dans VS Code (lance extension host)
# Ou dans la nouvelle fenêtre, tester via Copilot Chat

# 4. Smoke test
npm run smoke:local
npm run smoke:local:query -- "ta requête ici"
```

---

## ⚙️ Configuration Minimale

Dans `.vscode/settings.json` (workspace) ou user settings (global):

```json
{
  "deepResearch.backend": "local-first",
  "deepResearch.maxSources": 8,
  "deepResearch.enableReaderProxy": true
}
```

C'est tout ! 🎉 Le reste utilise les defaults.

---

## 🧪 Vérification Post-Installation

**Commandes disponibles:**
- `Ctrl+Shift+P` → `Deep Research: Show Jobs` ✅
- `Ctrl+Shift+P` → `Deep Research: Cancel Job` ✅
- Copilot Chat: `#deep_research ...` ✅

**Premiers rapports:**
- `.github/research_reports/research_*.md` ✅

**Pas de rapport ?**
```bash
# Créer le dossier
mkdir -p .github/research_reports
```

---

## 🚀 Cas d'Usage

### Recherche Académique
```
#deep_research Latest advances in Transformer architectures and scaling laws
```
→ arXiv, OpenAlex papers + Wikipedia reference

### Troubleshooting Ops
```
#deep_research Grafana Loki cardinality best practices
```
→ GitHub Issues, StackExchange, arXiv + official docs

### Exploration Technologique
```
#deep_research WebAssembly performance benchmarks vs native code
```
→ Repos, docs, papers, forums

---

## 📋 Fichiers Clés

Après installation, tu trouveras:

| Chemin | But |
|--------|-----|
| `.github/research_reports/` | Rapports de recherche générés |
| `.vscode/settings.json` | Configuration locale |
| `out/` (dev uniquement) | JavaScript compilé |
| `node_modules/` (dev) | Dépendances |

---

## 🆘 Problèmes Courants

| Problème | Solution |
|----------|----------|
| Extension pas visible | `code --install-extension ./copilot-deep-research-tool-0.0.1.vsix` + reload |
| Pas de rapport généré | Créer `.github/research_reports/` |
| Timeout réseau | Augmenter `deepResearch.requestTimeoutSeconds` à 15–20 |
| Reddit 403 | Attendu (rate limit), autres sources actives |
| Jina reader proxy down | Disable: `"deepResearch.enableReaderProxy": false` |

---

## 📞 Questions ?

- **Architecture ?** → Voir [README.md](./copilot-deep-research-tool/README.md)
- **Limitations ?** → Voir [FINAL_STATUS.md](./FINAL_STATUS.md)
- **Déploiement ?** → Voir [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- **Développement ?** → `npm run watch` + F5 (debug mode)

---

**Status:** ✅ Production-Ready | 📦 VSIX 50 KB | 🎯 Local-First + Cloud Fallback
