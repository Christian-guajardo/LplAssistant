# **🗺️ Feuille de Route et Master Prompt pour GitHub Copilot**

Ce document contient les instructions exactes à fournir à GitHub Copilot Chat (ou Claude/ChatGPT) pour développer l'extension VS Code intégrant Gemini Deep Research, en respectant l'architecture asynchrone définie précédemment.

## **🛠️ Comment utiliser ce document**

1. Créez un dossier vide pour votre extension et ouvrez-le dans VS Code.  
2. Ouvrez la vue **Copilot Chat**.  
3. Copiez-collez le **Master Prompt** ci-dessous pour initialiser le contexte de l'IA.  
4. Suivez ensuite les **Étapes de la Roadmap** une par une en demandant à Copilot de générer le code pour chaque étape spécifique.

## **💬 1\. Le "Master Prompt" (À copier-coller en premier)**

*Copiez le bloc de texte ci-dessous et envoyez-le à Copilot Chat pour lui donner les directives architecturales.*

**Prompt :**

"Agis en tant qu'ingénieur expert en développement d'extensions Visual Studio Code (TypeScript). Mon objectif est de créer une extension qui intègre l'agent 'Gemini Deep Research' de Google directement en tant qu'outil actionnable par GitHub Copilot.

Voici les contraintes architecturales strictes que tu devras respecter pour ce projet :

1. **Primitive d'extension :** Nous allons utiliser la récente API vscode.LanguageModelTool (déclaration dans package.json via contributes.languageModelTools). L'outil s'appellera gemini\_deep\_research.  
2. **L'API Google :** L'agent (modèle deep-research-pro-preview-12-2025) ne s'utilise pas avec generateContent. Il faut utiliser la nouvelle **API Interactions** (/v1beta/interactions).  
3. **Asynchronisme Extrême (Fire-and-Forget) :** La recherche prend de 15 à 45 minutes. La méthode invoke de notre outil NE DOIT PAS attendre la fin. Elle doit :  
   * Lancer la requête API avec le paramètre background: true.  
   * Retourner immédiatement un texte à Copilot disant que la recherche est lancée (avec l'ID de l'interaction).  
   * Lancer un gestionnaire de tâche en arrière-plan (polling) qui vérifie le statut de l'interaction toutes les 30 secondes.  
4. **Le 'Data Sink' :** Quand le statut passe à completed, l'extension doit récupérer le rapport texte, créer un nouveau fichier Markdown (.md) dans le workspace de l'utilisateur en utilisant vscode.workspace.fs, et l'ouvrir avec vscode.window.showTextDocument.  
5. **Sécurité :** La clé API Google doit être demandée à l'utilisateur et stockée UNIQUEMENT via vscode.SecretStorage.

Comprends-tu ces contraintes ? Si oui, réponds simplement 'Contexte assimilé. Prêt à commencer l'étape 1.' et attends mes prochaines instructions."

## **📍 2\. La Roadmap (À demander étape par étape)**

Une fois que Copilot a assimilé le Master Prompt, demandez-lui d'exécuter ces étapes l'une après l'autre. *Ne lui demandez pas tout d'un coup, le code serait trop complexe et risquerait de contenir des erreurs.*

### **Étape 1 : Initialisation et Manifeste**

**Prompt à envoyer :**

"Génère le contenu du fichier package.json pour notre extension. Assure-toi d'inclure les engines nécessaires pour les API récentes de VS Code, les dépendances de base (typescript, @types/vscode), la commande pour configurer la clé API, et surtout la section contributes.languageModelTools pour l'outil gemini\_deep\_research. Rédige une bonne modelDescription pour expliquer au LLM de Copilot qu'il doit utiliser cet outil pour les requêtes complexes nécessitant une recherche web approfondie asynchrone."

### **Étape 2 : Gestion de la Clé API (SecretStorage)**

**Prompt à envoyer :**

"Maintenant, crée un fichier src/secrets.ts. Implémente une classe ou des fonctions utilitaires pour gérer la clé API Google en utilisant vscode.SecretStorage. Il me faut une fonction pour vérifier si la clé existe, une fonction pour demander la clé à l'utilisateur via vscode.window.showInputBox (avec password: true), et une fonction pour la récupérer en mémoire."

### **Étape 3 : Le Client API Interactions de Google**

**Prompt à envoyer :**

"Crée un fichier src/geminiClient.ts. Implémente un client HTTP natif (utilise fetch natif de Node.js) pour interagir avec l'API Google Interactions.

Il me faut deux méthodes :

1. startDeepResearch(query: string, apiKey: string): Fait un POST vers /v1beta/interactions avec background: true et le modèle deep-research-pro-preview-12-2025. Retourne l'identifiant de l'interaction.  
2. checkInteractionStatus(interactionId: string, apiKey: string): Fait un GET pour vérifier le statut. Retourne le statut et, si c'est 'completed', retourne le texte contenu dans le tableau outputs."

### **Étape 4 : L'implémentation du Language Model Tool**

**Prompt à envoyer :**

"Génère le fichier src/researchTool.ts. Implémente l'interface vscode.LanguageModelTool.

Dans la méthode prepareInvocation, demande une simple confirmation à l'utilisateur.

Dans la méthode invoke :

* Récupère la clé API (via notre module secrets).  
* Lance startDeepResearch.  
* Déclenche le gestionnaire de tâche en arrière-plan (polling).  
* Retourne IMMÉDIATEMENT un vscode.LanguageModelToolResult à Copilot pour ne pas bloquer le chat, en lui indiquant que la tâche tourne en arrière-plan."

### **Étape 5 : Le Polling en arrière-plan et le Data Sink**

**Prompt à envoyer :**

"Crée le gestionnaire de tâche en arrière-plan (tu peux le mettre dans src/taskManager.ts ou à la suite du fichier précédent).

Implémente la boucle de polling qui appelle checkInteractionStatus toutes les 30 secondes. Utilise vscode.window.withProgress (en mode Notification) pour montrer que l'agent cherche.

Quand le statut est completed, implémente la logique du 'Data Sink' : utilise vscode.workspace.fs.writeFile pour écrire le résultat dans un fichier nommé deep-research-\<timestamp\>.md à la racine du workspace, puis ouvre-le automatiquement avec vscode.window.showTextDocument. Affiche une notification de succès."

### **Étape 6 : L'assemblage final (Point d'entrée)**

**Prompt à envoyer :**

"Enfin, génère le fichier src/extension.ts. Assemble tout. Enregistre les commandes (pour définir la clé API) et enregistre l'outil avec vscode.lm.registerTool. Ajoute les exports activate et deactivate nécessaires."

## **💡 Conseils pour le débogage avec Copilot**

* **Erreurs de Type (ts(2339)) :** L'API LanguageModelTool est très récente (fin 2024/2025). Si Copilot génère du code utilisant d'anciennes API d'IA de VS Code, corrigez-le en lui disant : *"Rappelle-toi, utilise l'API vscode.LanguageModelTool et vscode.lm.registerTool, l'interface requiert les méthodes prepareInvocation et invoke."*  
* **Timeout :** Si Copilot essaie de mettre un await sur la boucle de polling dans la méthode invoke, rappelez-lui la contrainte n°3 : la méthode invoke doit se terminer immédiatement pour libérer l'agent Copilot Chat.