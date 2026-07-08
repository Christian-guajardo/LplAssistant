# **Architecture et Intégration d'un Agent de Recherche Autonome dans l'Écosystème Visual Studio Code et GitHub Copilot**

## **1\. Analyse du Paradigme Actuel de l'Assistance au Développement**

L'évolution des environnements de développement intégrés (IDE) a été profondément marquée par l'intégration de l'intelligence artificielle. Des outils tels que GitHub Copilot ont transformé l'écriture de code, passant d'un simple système d'autocomplétion syntaxique à une assistance contextuelle avancée capable de générer des blocs entiers de logique métier, d'expliquer des architectures existantes et de proposer des refactorisations.1 Cependant, le paradigme actuel de ces assistants conversationnels repose fondamentalement sur un modèle synchrone et localisé. L'agent analyse le contexte immédiat—les fichiers ouverts, le terminal, la structure du projet—et génère une réponse quasi instantanée.2  
Cette architecture synchrone révèle ses limites lorsque les défis d'ingénierie nécessitent une exploration approfondie des connaissances externes. La prise de décision architecturale moderne exige souvent de comparer des documentations d'API disparates, d'analyser des retours d'expérience sur des forums spécialisés, d'étudier des rapports de marché ou d'évaluer les vulnérabilités de bibliothèques open source. Actuellement, un développeur confronté à ce besoin doit quitter son IDE, interrompre son flux de travail (le *flow*), et interagir manuellement avec des moteurs de recherche ou des interfaces d'agents autonomes externes.  
Bien que certains mécanismes permettent d'ancrer (via le *grounding*) les réponses d'un grand modèle de langage (LLM) avec des résultats de recherche web, cette approche reste superficielle.3 L'ancrage web classique, tel que le "Grounding with Google Search", fonctionne en exécutant une requête unique, en récupérant quelques extraits (snippets) de texte renvoyés par le moteur de recherche, puis en injectant ces fragments dans la fenêtre de contexte du LLM pour formuler une réponse immédiate.3 Cette méthode souffre d'un défaut structurel majeur pour la recherche complexe : le modèle ne navigue pas dans les pages, ne lit pas le contenu intégral et ne peut pas itérer ou ajuster sa stratégie de recherche en fonction de ce qu'il découvre en cours de route.4  
À l'inverse, l'agent "Gemini Deep Research" de Google opère selon une architecture radicalement différente. Il s'agit d'un flux de travail agentique asynchrone qui orchestre de manière autonome des boucles de planification, de recherche, de lecture, de raisonnement et de synthèse sur une longue durée.5 La volonté d'intégrer une telle puissance de recherche directement au sein de Visual Studio Code (VS Code), et plus spécifiquement en tant qu'outil actionnable de manière autonome par GitHub Copilot, représente une avancée majeure dans la conception des outils de développement. Cette intégration nécessite néanmoins de surmonter d'importants défis architecturaux, notamment la réconciliation entre l'attente de réponses synchrones de la part de l'interface de l'IDE et l'exécution asynchrone prolongée requise par la recherche autonome.  
Ce rapport détaille les méthodes, les architectures logicielles et les protocoles d'interface de programmation (API) requis pour construire une extension VS Code capable d'encapsuler l'agent Gemini Deep Research et de l'exposer nativement à GitHub Copilot Chat.

## **2\. Déconstruction de l'Agent Gemini Deep Research**

Pour intégrer efficacement cet outil, il est impératif d'en comprendre les fondations techniques et les contraintes opérationnelles. Le système Gemini Deep Research n'est pas un modèle de langage traditionnel qui génère un texte de manière séquentielle à partir d'une simple invite (prompt). Il s'agit d'un système composite, un agent optimisé pour les tâches de collecte et de synthèse de contexte sur un horizon temporel long.6

### **2.1 Le Modèle Sous-jacent et ses Capacités**

Le cœur de raisonnement de cet agent repose sur le modèle Gemini 3.1 Pro.5 Ce modèle a été spécifiquement entraîné pour maximiser la factualité, réduire les hallucinations et exécuter des tâches d'analyse critique nécessitant une précision absolue.6 Sur des bancs d'essai rigoureux évaluant la capacité à exécuter des recherches web complexes, tels que "Humanity's Last Exam" (HLE) ou "DeepSearchQA", l'agent atteint des scores de pointe, démontrant une compréhension nuancée de la topologie de l'information sur le web.6  
Le modèle Gemini 3.1 Pro est équipé d'une fenêtre de contexte massive pouvant atteindre plus d'un million de jetons (tokens).7 Cette capacité est cruciale car elle permet à l'agent de stocker temporairement le texte intégral de dizaines, voire de centaines de documents web lus au cours de son exploration, avant de procéder à la phase de synthèse finale.8

### **2.2 La Boucle Agentique de Recherche**

Lorsqu'une requête est soumise à l'agent (par exemple, "Analyse l'évolution des architectures de bases de données vectorielles entre 2024 et 2026 et compare leurs performances"), le déclenchement entraîne une boucle itérative 5 :

1. **Planification (Planning) :** L'agent décompose la question complexe en sous-sujets et formule un plan de recherche structuré.6  
2. **Recherche et Récupération (Search & Retrieve) :** Il génère de multiples requêtes de recherche spécifiques, interroge l'index web, identifie les URL pertinentes et procède à la lecture complète des pages cibles.6 Contrairement aux systèmes RAG (Retrieval-Augmented Generation) basiques, il navigue à travers des paysages d'information complexes.  
3. **Évaluation et Raisonnement (Iterate) :** L'agent évalue l'information récoltée par rapport à son plan initial. S'il identifie des lacunes dans ses connaissances ou des contradictions entre les sources, il génère de nouvelles requêtes pour affiner sa compréhension.5  
4. **Synthèse et Formatage (Synthesis) :** Une fois l'information jugée suffisante, l'agent consolide les données, rédige un rapport détaillé et, point fondamental pour la fiabilité, intègre des citations précises permettant de vérifier l'origine de chaque affirmation.6

### **2.3 Contraintes de Temps et de Format**

Cette profondeur d'analyse implique un coût temporel inhérent. Une tâche de recherche menée par cet agent peut nécessiter plusieurs minutes, avec une limite de temps d'exécution fixée à 60 minutes par tâche dans son état de préversion (preview) actuel.5 La latence n'est donc plus mesurée en millisecondes pour générer un bloc de code, mais en minutes pour produire un document analytique de plusieurs milliers de mots.5  
Le tableau suivant illustre la dichotomie fondamentale entre les systèmes de génération standards et l'agent de recherche profonde, justifiant des approches d'intégration totalement différentes dans un IDE.

| Caractéristique Architecturale | Modèles Gemini Standards (ex: Flash, Pro) | Agent Gemini Deep Research (deep-research-pro-preview-12-2025) |
| :---- | :---- | :---- |
| **Latence d'exécution** | Quelques secondes | De plusieurs minutes à une heure (Asynchrone/Background) 5 |
| **Mécanisme cognitif** | Génération directe ![][image1] Sortie textuelle | Planification ![][image1] Recherche ![][image1] Lecture ![][image1] Itération ![][image1] Synthèse 5 |
| **Mode de consommation** | Chatbots, extraction de données, autocomplétion | Analyse de marché, revues de littérature, tableaux comparatifs 5 |
| **Gestion de l'interface API** | Requêtes synchrones (generateContent) | Polling asynchrone (/interactions) avec suivi d'état 12 |

## **3\. Le Vecteur d'Intégration : L'API Google Interactions**

Historiquement, l'interaction avec les modèles de la famille Gemini s'effectuait via le point de terminaison (endpoint) generateContent. Bien que robuste pour des requêtes synchrones, cette approche obligeait le client (l'application appelante) à gérer l'intégralité de l'historique de la conversation et à maintenir des connexions HTTP ouvertes pendant la génération.14  
L'intégration de capacités agentiques telles que le Deep Research a forcé l'ingénierie de Google à introduire un nouveau paradigme de communication : l'API Interactions.14 Actuellement en version bêta publique, cette interface unifiée a été spécifiquement conçue pour simplifier la gestion de l'état, l'orchestration des outils et surtout, la gestion des tâches s'exécutant sur de longues durées (long-running tasks).12

### **3.1 Gestion de l'État Côté Serveur**

L'une des innovations majeures de l'API Interactions est sa capacité à gérer l'état de la conversation côté serveur de manière native. Dans une intégration VS Code classique, l'extension doit envoyer l'historique complet du chat à chaque nouvelle requête. Avec l'API Interactions, chaque échange génère un objet d'interaction sauvegardé sur l'infrastructure de Google (lorsque la propriété store=true est activée, ce qui est le comportement par défaut).12 Pour poursuivre une conversation ou interagir à nouveau avec un agent sur le même contexte, le client n'a qu'à transmettre le paramètre previous\_interaction\_id.12 Cela réduit drastiquement la charge utile (payload) du réseau et minimise les erreurs de gestion de contexte côté client.14

### **3.2 L'Exécution en Arrière-Plan et le Système de Polling**

L'accès à l'agent Deep Research est exclusif à cette nouvelle API ; il est impossible de l'invoquer via les méthodes generateContent traditionnelles.5 Le code d'agent spécifique à utiliser est deep-research-pro-preview-12-2025.8  
Étant donné la durée d'exécution du processus (qui peut s'étendre sur des dizaines de minutes), la requête initiale doit impérativement inclure le paramètre background: true (ou background=True selon le langage).5 Cette configuration modifie fondamentalement la nature de la requête HTTP. Au lieu de bloquer la connexion en attendant la fin de la recherche, l'API renvoie immédiatement une réponse contenant un identifiant d'interaction (ID) généré, accompagné d'un statut initial tel que processing ou started.17  
Dès lors, l'architecture logicielle de l'extension cliente doit implémenter une boucle de vérification (polling loop).5 Ce mécanisme interroge périodiquement l'endpoint de récupération (GET /v1beta/interactions/{interaction\_id}) en utilisant l'ID obtenu.12 À chaque vérification, le système inspecte le champ status. Si le statut indique completed, le client peut extraire le rapport final complet depuis le tableau outputs.5 Si une erreur critique survient durant les recherches, le statut passe à failed, permettant au client de gérer l'exception proprement.5

### **3.3 Considérations Financières et Quotas**

L'accès à cette API nécessite un compte développeur Google AI Studio ou une configuration Google Cloud Vertex AI.5 L'utilisation n'est pas couverte par les forfaits grand public (tels que Gemini Advanced ou Google One AI Premium) via les identifiants utilisateurs standards ; elle requiert une clé API (x-goog-api-key).20  
Le modèle économique repose sur une facturation à l'usage (pay-as-you-go) indexée sur la consommation de jetons (tokens) du modèle sous-jacent Gemini 3.1 Pro, à laquelle s'ajoutent les coûts inhérents aux outils mobilisés par l'agent, en particulier les requêtes massives adressées au moteur de recherche de Google durant son processus d'exploration.5 La planification financière d'une telle intégration doit prendre en compte le volume de contexte ingéré lors de la lecture des pages web complètes, qui peut rapidement faire grimper le nombre de jetons d'entrée (input tokens) traités.8

## **4\. Choix de la Modélisation d'Extensibilité dans Visual Studio Code**

Pour connecter ce puissant moteur asynchrone à l'expérience utilisateur fluide de VS Code et de GitHub Copilot, il faut sélectionner la bonne primitive d'extensibilité. Microsoft propose actuellement trois paradigmes principaux pour étendre les capacités d'intelligence artificielle au sein de son éditeur : les serveurs MCP, les Participants de Chat, et les Outils de Modèle de Langage (Language Model Tools).

### **4.1 Model Context Protocol (MCP)**

Le Model Context Protocol est un standard ouvert émergent conçu pour standardiser la manière dont les modèles d'IA interagissent avec les sources de données et les outils externes, indépendamment de l'IDE.21 Un développeur pourrait théoriquement créer un serveur MCP, fonctionnant comme un processus indépendant sur la machine locale, qui exposerait une fonction run\_deep\_research. VS Code agirait alors simplement comme un client MCP.21  
Cependant, cette approche "out-of-process" présente un inconvénient majeur dans le contexte présent : le serveur MCP ne tourne pas dans le contexte d'exécution direct de l'extension VS Code. De ce fait, il n'a pas un accès natif aux API internes de l'éditeur.22 Il devient très complexe de manipuler l'interface utilisateur (comme afficher des barres de progression natives), d'interagir avec le système de fichiers spécifique de l'espace de travail (Workspace) de manière fluide, ou d'utiliser le système de stockage sécurisé des secrets (SecretStorage) propre à VS Code pour gérer la clé API de Google de façon optimale.

### **4.2 Chat Participant API (@mention)**

La création d'un Chat Participant implique le développement d'un expert dédié que l'utilisateur invoque explicitement via une mention avec le symbole @, par exemple @researcher.23 Dans ce modèle, l'extension développée prend le contrôle total de la conversation, de la saisie de l'utilisateur à la restitution de la réponse.23  
L'avantage principal réside dans le contrôle absolu de l'expérience utilisateur et du formatage du flux de sortie (Markdown, arborescences de fichiers, boutons interactifs).23 Néanmoins, cela rompt la fluidité de l'expérience Copilot. L'utilisateur doit consciemment décider de quitter l'agent Copilot généraliste pour s'adresser au participant spécifique, empêchant ainsi l'agent IA de décider de lui-même quand une recherche approfondie est nécessaire pour accomplir une tâche de codage complexe.

### **4.3 Language Model Tool API (\#mention et Invocation Autonome)**

L'API Language Model Tool est l'approche la plus avancée et celle qui répond précisément à l'objectif de l'intégration : faire du Deep Research un outil actionnable directement *par* Copilot.22  
Au lieu de gérer la conversation de bout en bout, l'extension enregistre une fonction spécifique (un outil) auprès du système central.22 Cet outil est accompagné d'un schéma JSON décrivant précisément ses capacités, ses paramètres d'entrée et ses limites.22 Lorsque l'utilisateur discute avec l'agent par défaut de Copilot (en mode "Agent"), l'orchestrateur (le grand modèle de langage pilotant Copilot, tel que Claude 3.5 Sonnet ou GPT-4o) analyse la requête de l'utilisateur et évalue les outils à sa disposition.22  
Si la requête exige une recherche contextuelle vaste que le modèle ne possède pas dans ses poids d'entraînement, le modèle décide *de manière autonome* de générer une requête pour appeler l'outil gemini\_deep\_research.22 L'IDE transmet alors les paramètres générés par l'orchestrateur à la fonction d'implémentation de l'extension, qui exécute l'appel à l'API Google, puis renvoie les résultats à l'orchestrateur.22 L'orchestrateur synthétise enfin ces résultats pour fournir la réponse finale à l'utilisateur, intégrant les fruits de la recherche web dans des recommandations de code spécifiques au projet.22 De plus, l'utilisateur conserve la possibilité de forcer l'utilisation de l'outil en utilisant la notation \# (ex: \#deep\_research) directement dans l'interface de chat.22  
C'est cette troisième architecture qui doit être retenue pour garantir l'intégration la plus transparente et symbiotique au sein du flux de développement.

## **5\. Ingénierie de l'Outil de Modèle de Langage (Language Model Tool)**

La construction de l'outil gemini\_deep\_research s'articule autour de trois axes de développement au sein d'une extension VS Code : la déclaration statique dans le manifeste de l'extension, l'enregistrement dynamique dans le cycle de vie de l'IDE, et la gestion épineuse de l'asynchronisme imposée par la nature même du Deep Research.

### **5.1 Déclaration Statique et Ingénierie de Prompt**

La première étape consiste à déclarer l'outil dans le fichier package.json de l'extension, sous la clé de contribution contributes.languageModelTools.22 Cette déclaration ne sert pas uniquement à l'interface utilisateur ; elle est ingérée par l'orchestrateur LLM de Copilot. La description fournie constitue un véritable exercice d'ingénierie de prompt (prompt engineering).

JSON  
"contributes": {  
  "languageModelTools":  
      }  
    }  
  \]  
}

La propriété modelDescription est fondamentale.22 Elle informe le LLM de Copilot de la nature de la tâche et, surtout, de sa latence extrême. Sans un avertissement explicite concernant la durée d'exécution et le périmètre d'utilisation, l'orchestrateur risquerait de déléguer des tâches triviales à l'agent Deep Research, provoquant des délais d'attente inacceptables pour l'utilisateur. Le inputSchema définit formellement que l'outil attend une chaîne de caractères nommée research\_query, garantissant que le LLM formatera correctement son appel de fonction.22

### **5.2 Implémentation TypeScript et Enregistrement**

Dans le code source de l'extension (généralement src/extension.ts), l'outil doit être instancié et enregistré lors de l'événement d'activation de l'extension en utilisant l'API vscode.lm.registerTool.22  
L'implémentation exige la création d'une classe qui satisfait l'interface vscode.LanguageModelTool\<T\>. Cette interface impose la définition de deux méthodes principales : prepareInvocation et invoke.22  
La méthode prepareInvocation est cruciale pour la sécurité et la transparence. Avant que l'outil ne soit exécuté et ne transmette potentiellement des données du projet vers les serveurs de Google, l'IDE présente une boîte de dialogue de confirmation à l'utilisateur.22 Cette méthode permet de personnaliser le message de consentement, en affichant par exemple la requête que le LLM s'apprête à envoyer à l'agent de recherche.22

### **5.3 Résolution du Paradoxe de l'Asynchronisme Extrême**

Le véritable défi technique réside dans l'implémentation de la méthode invoke. L'architecture de Copilot s'attend à ce qu'un appel d'outil se résolve dans un délai raisonnable (généralement quelques secondes à quelques minutes) pour poursuivre son traitement.22 Bloquer l'exécution de la fonction invoke avec une promesse asynchrone (await) pendant 30 ou 60 minutes pendant que l'agent Google travaille entraînera inévitablement un dysfonctionnement de l'agent Copilot, qui peut se bloquer indéfiniment ou renvoyer une erreur de dépassement de délai ("timeout") ou de réponse tronquée.25  
Pour contourner ce goulot d'étranglement architectural, deux motifs de conception (design patterns) peuvent être envisagés.

#### **Motif 1 : La Boucle de Polling Interne avec Rétroaction Visuelle**

Si l'objectif est de maintenir le flux au sein d'une seule et même requête Copilot, l'extension doit implémenter une boucle de vérification (polling) tout en signalant activement à l'IDE que le processus est toujours en cours de manière légitime.  
Dans la méthode invoke, le code émet une requête POST à l'API Interactions de Google avec la directive background: true.17 L'API répond quasi instantanément avec l'identifiant de l'interaction. Dès réception, l'extension déclenche l'API de Progression de VS Code (vscode.window.withProgress) pour afficher un indicateur de chargement discret (location ProgressLocation.Notification ou dans la barre d'état) avertissant l'utilisateur que l'agent Deep Research rassemble des informations.27  
Parallèlement, la méthode invoke initie une boucle while asynchrone, entrecoupée de pauses (setTimeout), qui interroge l'endpoint GET de l'API Interactions avec l'ID obtenu.17 Il est primordial de lier cette boucle au jeton d'annulation fourni par VS Code (token: vscode.CancellationToken). Si l'utilisateur clique sur le bouton d'arrêt dans l'interface de chat, token.isCancellationRequested deviendra vrai, permettant à la boucle de s'interrompre proprement, d'annuler éventuellement l'interaction côté serveur Google si l'API le permet, et d'éviter des fuites de mémoire.22  
Bien que cette méthode fournisse un retour visuel, maintenir le processus de l'outil ouvert pendant 45 minutes reste risqué vis-à-vis des délais d'attente internes des moteurs de routage LLM de Microsoft.25

#### **Motif 2 : L'Architecture de Tâches Distribuées (Fire-and-Forget / Sink)**

L'approche architecturale la plus résiliente consiste à découpler l'initialisation de la recherche de la restitution de ses résultats, adaptant ainsi l'outil aux limitations de contexte et de temps de l'IDE. Plutôt que d'attendre la fin de la recherche dans la méthode invoke, l'outil signale immédiatement à Copilot que la tâche a été déléguée.

1. La méthode invoke reçoit la requête de Copilot.  
2. L'extension effectue l'appel initial à l'API Interactions pour lancer le Deep Research (background: true) et récupère l'identifiant.17  
3. Immédiatement, la méthode invoke retourne un objet vscode.LanguageModelToolResult à Copilot contenant un texte tel que : *"La recherche approfondie a été lancée avec succès sur les serveurs de Google (ID: ia\_xyz123). Cette opération prendra entre 15 et 45 minutes. Informez l'utilisateur qu'il peut continuer à coder normalement et qu'il sera notifié dès la fin de l'analyse."* 22  
4. Copilot relaie cette information à l'utilisateur et clôt le cycle de chat en cours. L'interface redevient entièrement disponible.  
5. En arrière-plan, en dehors du cycle de vie strict de l'outil, un service de l'extension (un *Background Task Manager*) effectue le polling de l'API Google de manière totalement asynchrone.18  
6. Lorsque le statut de l'interaction passe à completed, le service de l'extension prend le relais pour gérer les données volumineuses de la réponse.

### **5.4 La Gestion du "Data Sink" et de la Surcharge Cognitive**

La dernière étape du processus de recherche profonde pose un défi majeur lié à l'ergonomie et à la gestion de la mémoire contextuelle. Le rapport généré par Gemini Deep Research est exhaustif, structuré, lourdement sourcé par des citations, et peut s'étaler sur des milliers de mots.6  
Si ce document monumental était réinjecté directement dans le flux de la fenêtre de discussion Copilot Chat, il saturerait la mémoire contextuelle locale du modèle d'orchestration.11 Le LLM de base oublierait instantanément les instructions antérieures de l'utilisateur, et l'interface utilisateur du chat deviendrait illisible, noyée sous des pages de texte Markdown.  
La solution architecturale optimale nécessite la création d'un "Data Sink" (puits de données) local. Lorsque la boucle de vérification asynchrone de l'extension détecte l'achèvement du processus et récupère la charge utile (payload) massive depuis l'API, le comportement suivant doit être scripté :

1. L'extension utilise l'API native du système de fichiers de l'éditeur (vscode.workspace.fs) pour créer un nouveau fichier Markdown dans le répertoire du projet, par exemple .github/research\_reports/analyse\_vector\_db\_2026.md.  
2. L'intégralité de la chaîne de caractères renvoyée par interaction.outputs\[-1\].text y est inscrite.17  
3. L'extension fait appel à vscode.window.showTextDocument pour ouvrir automatiquement ce fichier de rapport riche dans un nouvel onglet de l'éditeur, offrant à l'utilisateur un environnement de lecture confortable.  
4. L'extension envoie une notification push native via vscode.window.showInformationMessage, signalant que la recherche est terminée et que le fichier a été généré, invitant le développeur à poser des questions supplémentaires à Copilot en mentionnant ce nouveau fichier en contexte (ex: @workspace en te basant sur le fichier \#analyse\_vector\_db\_2026.md, mets à jour ma classe de connexion).

Ce motif de conception préserve la fenêtre de contexte de Copilot, maintient la réactivité de l'IDE, et stocke la connaissance acquise de manière persistante et versionnable au sein du dépôt de code de l'utilisateur.

## **6\. Gouvernance Sécuritaire des Accès API**

Un composant non négligeable de cette architecture concerne la sécurité opérationnelle. L'invocation de l'API Interactions nécessite l'authentification des requêtes via un en-tête HTTP contenant une clé API (x-goog-api-key: $GEMINI\_API\_KEY) obtenue depuis la console Google AI Studio.12  
Étant donné que l'utilisateur doit fournir sa propre clé (modèle dit "Bring Your Own Key" ou BYOK), la manière dont l'extension stocke cette information est critique.

### **6.1 L'Anti-Modèle du Stockage en Clair**

Une implémentation naïve consisterait à demander à l'utilisateur de placer sa clé dans un fichier d'environnement local (.env) ou de la définir dans les paramètres globaux de l'éditeur via le fichier settings.json ou l'interface de configuration (vscode.WorkspaceConfiguration).32 Cette approche constitue une vulnérabilité de sécurité sévère. Les données sauvegardées via ces méthodes, ou même via l'objet d'état de l'extension (vscode.ExtensionContext.workspaceState ou globalState), sont persistées en clair (plaintext) sur le système de fichiers local.33 De plus, les données inscrites dans les paramètres de VS Code sont susceptibles d'être synchronisées via la fonctionnalité "Settings Sync" et transmises à travers les réseaux, augmentant considérablement la surface d'attaque.32  
La compromission d'une clé API Google peut entraîner des conséquences financières désastreuses, étant donné le modèle de facturation à l'usage des modèles avancés comme Gemini 3.1 Pro, exacerbé par le coût potentiellement élevé des tâches de recherche profondes (qui consomment de vastes quantités de jetons en ingérant des pages web entières).8

### **6.2 Exploitation de l'Interface SecretStorage**

La méthodologie sécurisée imposée par les directives d'extensibilité de Microsoft repose sur l'utilisation exclusive de l'API SecretStorage (vscode.SecretStorage) fournie par le contexte de l'extension.33  
Le système SecretStorage ne réinvente pas la cryptographie ; il s'interface de manière transparente avec les gestionnaires de trousseaux d'accès natifs du système d'exploitation hôte. Sur Windows, la clé sera chiffrée et protégée par le Windows Credential Manager ; sur macOS, elle résidera de manière sécurisée dans le Keychain ; et sur les distributions Linux, elle exploitera libsecret ou gnome-keyring.34  
Lors de sa première utilisation, si l'outil détecte l'absence de la clé, l'extension doit invoquer une boîte de dialogue de saisie vscode.window.showInputBox configurée avec la propriété password: true pour masquer les caractères frappés au clavier.38 La valeur est alors immédiatement transférée vers le trousseau du système via la méthode asynchrone context.secrets.store('gemini\_api\_key', keyInput).33 Par la suite, chaque fois que la méthode invoke du Language Model Tool est déclenchée par Copilot, la clé est extraite silencieusement et de manière éphémère en mémoire vive à l'aide de context.secrets.get('gemini\_api\_key') pour construire l'en-tête HTTP.33 Ce cloisonnement garantit que la clé ne transite jamais sous une forme non protégée.

## **7\. Modèles Économiques et Alternatives Open Source**

L'implémentation de cette architecture nécessite une évaluation des coûts pour l'utilisateur final. Bien que le modèle économique précis évolue, l'agent Deep Research s'appuie sur la tarification standard du modèle gemini-3.1-pro-preview et des outils associés.5  
Les tâches de l'agent ne consomment pas seulement des jetons de sortie (output tokens) pour la rédaction du rapport. Elles génèrent d'énormes volumes de jetons d'entrée (input tokens) causés par la lecture du contenu brut des URL extraites au fil des itérations de recherche.8 De surcroît, le processus exploite la fonctionnalité de recherche web (Grounding with Google Search), qui peut engendrer des surcoûts spécifiques basés sur le volume de requêtes de recherche émises de manière autonome par le système (les rapports indiquent souvent des facturations calculées par tranche de 1000 requêtes de recherche ancrées, au-delà d'un volume de base gratuit).35 Les utilisateurs bénéficiant d'un abonnement Google AI Ultra (ou équivalent pro) devront s'assurer que leurs limites de facturation de plateforme API (Google Cloud Vertex AI ou AI Studio) sont adaptées à la voracité en jetons de ce mode opératoire.

### **7.1 L'Alternative de Construction Architecturale (ReAct avec LangGraph)**

Il est pertinent de noter qu'en cas de restrictions sur l'utilisation d'API bêta propriétaires, ou si l'utilisateur souhaite un contrôle granulaire absolu sur la nature des moteurs de recherche employés, les mêmes capacités de l'agent peuvent être reconstruites via l'orchestration de modèles open source.31  
L'utilisation de frameworks tels que LangGraph (qui modélise les flux d'agents sous forme de graphes cycliques gérant l'état de manière explicite) ou l'Agent Development Kit (ADK) permet de simuler la boucle de recherche profonde.40 L'architecte peut développer un système distribué où un nœud de planification évalue le prompt, un nœud d'action lance des requêtes vers une API de scraping (comme SerpAPI), injecte le code HTML converti en Markdown dans un contexte partagé, et boucle jusqu'à ce qu'un nœud d'évaluation estime l'information suffisante.42 Bien que cette voie exige une ingénierie de pointe pour la gestion des limites de contexte et la prévention des boucles infinies de l'agent, elle s'intégrerait exactement selon la même logique au sein de l'extension VS Code via l'API Language Model Tool.

## **8\. Conclusion**

La convergence entre les environnements de développement intégrés et l'intelligence artificielle ne se limite plus à la simple suggestion de code localisée. En connectant l'IDE à l'infrastructure d'agents autonomes, le flux de travail de l'ingénierie logicielle s'étend pour inclure la recherche analytique asynchrone approfondie.  
L'intégration de l'agent "Gemini Deep Research" au sein de Visual Studio Code et de GitHub Copilot Chat est tout à fait réalisable en orchestrant les API appropriées. Le fondement de cette architecture requiert l'abandon de l'API de génération classique au profit de la nouvelle API Interactions de Google, seule capable de gérer la persistance de l'état côté serveur et le traitement en arrière-plan asynchrone indispensables au code agent deep-research-pro-preview-12-2025.12  
Côté éditeur, le choix architectural doit se porter fermement sur le modèle d'extensibilité "Language Model Tool" (API vscode.LanguageModelTool) plutôt que sur la création d'un "Chat Participant" dédié.22 Ce choix confère au grand modèle de langage sous-jacent de Copilot la capacité de déclencher la recherche de manière autonome, tout en laissant à l'utilisateur la possibilité de la forcer avec la notation \#.22  
Toutefois, la réussite de cette intégration dépend de l'ingénierie de la gestion de l'état asynchrone pour contourner les délais d'attente (timeouts) inhérents à Copilot. L'adoption d'un paradigme de "Data Sink", consistant à écrire directement les volumineux rapports générés sur le système de fichiers (vscode.workspace.fs) au format Markdown, plutôt que de polluer la mémoire de contexte de l'agent conversationnel, assure la robustesse du système final. En conjuguant cette architecture avec un stockage inébranlable des identifiants cryptés via l'interface native SecretStorage de VS Code 33, le développeur bénéficie d'une solution unifiée où la synthèse globale des connaissances du web soutient directement la logique de la base de code locale, le tout sans jamais quitter le contexte de son espace de travail.

#### **Sources des citations**

1. GitHub \- microsoft/vscode-copilot-chat: Copilot Chat extension for VS Code, consulté le mars 7, 2026, [https://github.com/microsoft/vscode-copilot-chat](https://github.com/microsoft/vscode-copilot-chat)  
2. Chat overview \- Visual Studio Code, consulté le mars 7, 2026, [https://code.visualstudio.com/docs/copilot/chat/copilot-chat](https://code.visualstudio.com/docs/copilot/chat/copilot-chat)  
3. Grounding with Google Search | Gemini API, consulté le mars 7, 2026, [https://ai.google.dev/gemini-api/docs/google-search](https://ai.google.dev/gemini-api/docs/google-search)  
4. Gemini 3 Pro Search functionality and Deep Research is by far the worst of any AI Platform, consulté le mars 7, 2026, [https://www.reddit.com/r/Bard/comments/1p3zapz/gemini\_3\_pro\_search\_functionality\_and\_deep/](https://www.reddit.com/r/Bard/comments/1p3zapz/gemini_3_pro_search_functionality_and_deep/)  
5. Gemini Deep Research Agent | Gemini API | Google AI for Developers, consulté le mars 7, 2026, [https://ai.google.dev/gemini-api/docs/deep-research](https://ai.google.dev/gemini-api/docs/deep-research)  
6. Build with Gemini Deep Research \- Google Blog, consulté le mars 7, 2026, [https://blog.google/innovation-and-ai/technology/developers-tools/deep-research-agent-gemini-api/](https://blog.google/innovation-and-ai/technology/developers-tools/deep-research-agent-gemini-api/)  
7. Gemini 3 Developer Guide | Gemini API \- Google AI for Developers, consulté le mars 7, 2026, [https://ai.google.dev/gemini-api/docs/gemini-3](https://ai.google.dev/gemini-api/docs/gemini-3)  
8. Deep Research preview | Gemini API | Google AI for Developers, consulté le mars 7, 2026, [https://ai.google.dev/gemini-api/docs/models/deep-research-pro-preview-12-2025](https://ai.google.dev/gemini-api/docs/models/deep-research-pro-preview-12-2025)  
9. Vertex AI Platform | Google Cloud, consulté le mars 7, 2026, [https://cloud.google.com/vertex-ai](https://cloud.google.com/vertex-ai)  
10. Get reports with Deep Research | Gemini Enterprise \- Google Cloud Documentation, consulté le mars 7, 2026, [https://docs.cloud.google.com/gemini/enterprise/docs/research-assistant](https://docs.cloud.google.com/gemini/enterprise/docs/research-assistant)  
11. I built a local deep research agent \- here's how it works : r/LocalLLM \- Reddit, consulté le mars 7, 2026, [https://www.reddit.com/r/LocalLLM/comments/1jzbkev/i\_built\_a\_local\_deep\_research\_agent\_heres\_how\_it/](https://www.reddit.com/r/LocalLLM/comments/1jzbkev/i_built_a_local_deep_research_agent_heres_how_it/)  
12. Interactions API | Gemini API \- Google AI for Developers, consulté le mars 7, 2026, [https://ai.google.dev/gemini-api/docs/interactions](https://ai.google.dev/gemini-api/docs/interactions)  
13. How to Use the Gemini Deep Research API in Production | by Karl Weinmeister \- Medium, consulté le mars 7, 2026, [https://medium.com/google-cloud/how-to-use-the-gemini-deep-research-api-in-production-978055873a39](https://medium.com/google-cloud/how-to-use-the-gemini-deep-research-api-in-production-978055873a39)  
14. Interactions API: A unified foundation for models and agents \- Google Blog, consulté le mars 7, 2026, [https://blog.google/innovation-and-ai/technology/developers-tools/interactions-api/](https://blog.google/innovation-and-ai/technology/developers-tools/interactions-api/)  
15. Building agents with the ADK and the new Interactions API \- Google Developers Blog, consulté le mars 7, 2026, [https://developers.googleblog.com/building-agents-with-the-adk-and-the-new-interactions-api/](https://developers.googleblog.com/building-agents-with-the-adk-and-the-new-interactions-api/)  
16. gemini-skills/skills/gemini-interactions-api/SKILL.md at main · google-gemini/gemini-skills \- GitHub, consulté le mars 7, 2026, [https://github.com/google-gemini/gemini-skills/blob/main/skills/gemini-interactions-api/SKILL.md](https://github.com/google-gemini/gemini-skills/blob/main/skills/gemini-interactions-api/SKILL.md)  
17. Google Gemini Just Got WAY Smarter\! | by Code and Bird \- Medium, consulté le mars 7, 2026, [https://medium.com/@codeandbird/google-gemini-just-got-way-smarter-27ef9ef7c82c](https://medium.com/@codeandbird/google-gemini-just-got-way-smarter-27ef9ef7c82c)  
18. Getting Started with Gemini Deep Research API \- Philschmid, consulté le mars 7, 2026, [https://www.philschmid.de/gemini-deep-research-getting-started](https://www.philschmid.de/gemini-deep-research-getting-started)  
19. Vertex AI Studio | Google Cloud, consulté le mars 7, 2026, [https://cloud.google.com/generative-ai-studio](https://cloud.google.com/generative-ai-studio)  
20. Gemini API reference | Google AI for Developers, consulté le mars 7, 2026, [https://ai.google.dev/api](https://ai.google.dev/api)  
21. I am still confused on the difference between Model Context Protocol vs Tool Calling (Function Calling); What are the limitations and boundaries of both? : r/mcp \- Reddit, consulté le mars 7, 2026, [https://www.reddit.com/r/mcp/comments/1lxg4qx/i\_am\_still\_confused\_on\_the\_difference\_between/](https://www.reddit.com/r/mcp/comments/1lxg4qx/i_am_still_confused_on_the_difference_between/)  
22. Language Model Tool API | Visual Studio Code Extension API, consulté le mars 7, 2026, [https://code.visualstudio.com/api/extension-guides/ai/tools](https://code.visualstudio.com/api/extension-guides/ai/tools)  
23. Chat Participant API \- Visual Studio Code, consulté le mars 7, 2026, [https://code.visualstudio.com/api/extension-guides/ai/chat](https://code.visualstudio.com/api/extension-guides/ai/chat)  
24. Use tools with agents \- Visual Studio Code, consulté le mars 7, 2026, [https://code.visualstudio.com/docs/copilot/agents/agent-tools](https://code.visualstudio.com/docs/copilot/agents/agent-tools)  
25. Response is Truncated because it is too long · community · Discussion \#170232 \- GitHub, consulté le mars 7, 2026, [https://github.com/orgs/community/discussions/170232](https://github.com/orgs/community/discussions/170232)  
26. Copilot does not detect terminal command has completed or is not getting terminal output · community · Discussion \#161238 \- GitHub, consulté le mars 7, 2026, [https://github.com/orgs/community/discussions/161238](https://github.com/orgs/community/discussions/161238)  
27. Status Bar | Visual Studio Code Extension API, consulté le mars 7, 2026, [https://code.visualstudio.com/api/ux-guidelines/status-bar](https://code.visualstudio.com/api/ux-guidelines/status-bar)  
28. Notifications | Visual Studio Code Extension API, consulté le mars 7, 2026, [https://code.visualstudio.com/api/ux-guidelines/notifications](https://code.visualstudio.com/api/ux-guidelines/notifications)  
29. Extension Capabilities Overview \- Visual Studio Code, consulté le mars 7, 2026, [https://code.visualstudio.com/api/extension-capabilities/overview](https://code.visualstudio.com/api/extension-capabilities/overview)  
30. VS Code API | Visual Studio Code Extension API, consulté le mars 7, 2026, [https://code.visualstudio.com/api/references/vscode-api](https://code.visualstudio.com/api/references/vscode-api)  
31. I built Open Source Deep Research \- here's how it works : r/LLMDevs \- Reddit, consulté le mars 7, 2026, [https://www.reddit.com/r/LLMDevs/comments/1jpfa8f/i\_built\_open\_source\_deep\_research\_heres\_how\_it/](https://www.reddit.com/r/LLMDevs/comments/1jpfa8f/i_built_open_source_deep_research_heres_how_it/)  
32. How to store secrets (license keys, passwords etc.) in VS Code extensions configuration?, consulté le mars 7, 2026, [https://www.reddit.com/r/vscode/comments/iggfsk/how\_to\_store\_secrets\_license\_keys\_passwords\_etc/](https://www.reddit.com/r/vscode/comments/iggfsk/how_to_store_secrets_license_keys_passwords_etc/)  
33. Supporting Remote Development and GitHub Codespaces | Visual Studio Code Extension API, consulté le mars 7, 2026, [https://code.visualstudio.com/api/advanced-topics/remote-extensions](https://code.visualstudio.com/api/advanced-topics/remote-extensions)  
34. Why Every Developer's API Keys Are Probably in the Wrong Place And how a VS Code Extension Finally… \- Medium, consulté le mars 7, 2026, [https://medium.com/@dingersandks/why-every-developers-api-keys-are-probably-in-the-wrong-place-and-how-a-vs-code-extension-finally-c966d081d132](https://medium.com/@dingersandks/why-every-developers-api-keys-are-probably-in-the-wrong-place-and-how-a-vs-code-extension-finally-c966d081d132)  
35. Gemini API Pricing and Quotas: Complete 2026 Guide with Cost Calculator, consulté le mars 7, 2026, [https://aifreeapi.com/en/posts/gemini-api-pricing-and-quotas](https://aifreeapi.com/en/posts/gemini-api-pricing-and-quotas)  
36. Gemini Developer API pricing, consulté le mars 7, 2026, [https://ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)  
37. VSCode Extension Authoring \- Storage for app secrets? \- Stack Overflow, consulté le mars 7, 2026, [https://stackoverflow.com/questions/45293157/vscode-extension-authoring-storage-for-app-secrets](https://stackoverflow.com/questions/45293157/vscode-extension-authoring-storage-for-app-secrets)  
38. How to use SecretStorage in your VSCode extensions \- DEV Community, consulté le mars 7, 2026, [https://dev.to/kompotkot/how-to-use-secretstorage-in-your-vscode-extensions-2hco](https://dev.to/kompotkot/how-to-use-secretstorage-in-your-vscode-extensions-2hco)  
39. Gemini API and Google AI Studio now offer Grounding with Google Search, consulté le mars 7, 2026, [https://developers.googleblog.com/en/gemini-api-and-ai-studio-now-offer-grounding-with-google-search/](https://developers.googleblog.com/en/gemini-api-and-ai-studio-now-offer-grounding-with-google-search/)  
40. Building agents with Google Gemini and open source frameworks, consulté le mars 7, 2026, [https://developers.googleblog.com/building-agents-google-gemini-open-source-frameworks/](https://developers.googleblog.com/building-agents-google-gemini-open-source-frameworks/)  
41. Build a Gemini Full-Stack Research Agent with Agent Development Kit | by Debi Cabrera \- Googler | Google Cloud \- Medium, consulté le mars 7, 2026, [https://medium.com/google-cloud/build-a-gemini-full-stack-research-agent-with-agent-development-kit-11192e8e9165](https://medium.com/google-cloud/build-a-gemini-full-stack-research-agent-with-agent-development-kit-11192e8e9165)  
42. LangGraph 101: Let's Build A Deep Research Agent | Towards Data Science, consulté le mars 7, 2026, [https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/](https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/)  
43. ReAct agent from scratch with Gemini and LangGraph \- Google AI for Developers, consulté le mars 7, 2026, [https://ai.google.dev/gemini-api/docs/langgraph-example](https://ai.google.dev/gemini-api/docs/langgraph-example)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAWCAYAAADNX8xBAAAAfUlEQVR4XmNgGAWjgCQgC8TdQMyBLkEq4AfizUCsiS5BDiiHYoqBGBDvB2IzdAkeIJYkAwcDcRIQczJAQQUQPyIDPwPiV0Acz0AB4AbihQxIriEHsADxVCAuQ5cgBYAMAbnEA12CVCDNAElHIugSpAJWIBYCYkZ0iVGAHwAA5R4XcDpvb24AAAAASUVORK5CYII=>