### Rapport d'Implémentation : Conception d'un Assistant Personnel "Jarvis" Local et Haute Performance

#### 1. Vision Stratégique et Objectifs du Système

Le basculement d'une IA basée sur le cloud vers une architecture locale n'est pas une simple préférence technique, c'est un impératif de souveraineté et de performance. Pour un assistant de type "Jarvis", la confidentialité des données personnelles est non négociable, et seule une exécution  *on-premise*  garantit l'absence de fuites vers des tiers. Plus encore, l'architecture locale permet d'éliminer la gigue réseau pour atteindre des latences sub-100ms, transformant l'expérience d'un "chatbot" poussif en un assistant réactif en temps réel. L'optimisation d'une stack logicielle en C++ est le levier fondamental de cette transformation. Contrairement aux environnements de haut niveau, le développement "proche du métal" permet une gestion chirurgicale des ressources, minimisant l'overhead et garantissant un déterminisme indispensable au traitement de flux audio et tensoriels synchrones. Cette excellence logicielle est toutefois vaine sans un bus matériel capable de soutenir des transferts de données déterministes à haute fréquence.

#### 2. Le Cerveau Central : Infrastructure de Calcul et Inférence IA

Le nœud central agit comme le concentrateur de puissance tensorielle. Son rôle est de minimiser le  *Time To First Token*  (TTFT) en traitant les modèles de langage (LLM) et de reconnaissance vocale (STT) avec une bande passante mémoire maximale.

| Caractéristique | NVIDIA Jetson Orin Nano | Raspberry Pi 5 + Hailo-8L |
| ------ | ------ | ------ |
| **Puissance IA (TOPS)** | Jusqu'à 40 TOPS | 13 TOPS |
| **Chemin Mémoire** | **Architecture Zero-copy**  (RAM unifiée) | Bus PCIe (Overhead de transfert) |
| **Gestion des données** | Accès direct via kernels CUDA | Copie CPU vers NPU via PCIe |
| **Écosystème** | Support natif CUDA / TensorRT | APIs constructeur (Hailo) |
| **Coût indicatif** | ~620 € (Kit complet) | ~237 € (Kit complet) |

L'architecture  *Zero-copy*  du Jetson Orin Nano constitue l'avantage comparatif décisif. En partageant la même mémoire physique entre le CPU et le GPU, on élimine les transferts de données redondants via le bus PCIe. Cette configuration permet aux kernels CUDA de manipuler les tenseurs directement là où ils sont décodés, garantissant une latence minimale. Sur un Raspberry Pi, l'overhead induit par le transfert vers l'accélérateur Hailo crée un goulot d'étranglement structurel qui bride l'interactivité. Ce choix matériel est le socle sur lequel repose l'ensemble des optimisations logicielles ultérieures.

#### 3. Le Système Nerveux Périphérique : Acquisition et Satellites IoT

Une couverture sensorielle domestique complète exige une architecture distribuée. Le cerveau central ne peut capter l'audio de manière optimale s'il est enfermé dans une baie technique ; il nécessite des satellites périphériques agissant comme ses "oreilles". L' **ESP32**  est le microcontrôleur pivot de ce système nerveux. Il gère de manière autonome le  *Wake Word*  via des modèles  **TinyML**  ultra-légers. Une fois le mot-clé détecté, l'ESP32 initie un streaming audio brut au format  **PCM via UDP** . Le choix de l'UDP est critique : l'absence de handshakes lourds et de mécanismes de retransmission garantit une latence constante et une gigue ( *jitter* ) minimale, facteurs essentiels pour la précision du moteur STT central.Pour la capture, le microphone  **ICS-43434**  surpasse l'INMP441 grâce à un rapport signal/bruit (SNR) élevé et une plage dynamique étendue. Ces caractéristiques sont vitales pour implémenter un  *beamforming*  efficace, permettant de distinguer la voix de l'utilisateur du bruit ambiant. En retour, Jarvis interagit avec l'environnement via l'API REST de  **Home Assistant**  ou des serveurs  **MCP**  (Model Context Protocol), synchronisant les événements périphériques avec la boucle de traitement centrale de manière asynchrone et non-bloquante.

#### 4. Architecture Logicielle et Optimisation de la Mémoire Vive

Le déterminisme logiciel est la clé d'un système "toujours à l'écoute". Chaque cycle CPU gaspillé est une milliseconde de latence ajoutée à la réponse de l'assistant. La stack recommandée repose sur le  **C/C++**  avec  **llama.cpp**  et  **whisper.cpp** . L'exécution "proche du métal" permet des optimisations de mémoire avancées :

* **Pools pré-alloués & Slab Allocators :**  L'utilisation d'allocateurs personnalisés prévient la fragmentation mémoire et les pics de latence lors de l'inférence.  
* **KV Caching (Key-Value Caching) :**  Indispensable pour l'inférence autorégressive. En stockant les calculs intermédiaires des tokens précédents dans la VRAM, on évite de recalculer l'intégralité du contexte à chaque nouveau mot généré.  
* **Multi-Token Prediction (Architecture Writer/Editor) :**  À l'instar de modèles comme GLM 5.2, l'implémentation d'une génération parallélisée où un "junior writer" propose plusieurs tokens validés instantanément par un "editor" permet d'accélérer drastiquement le débit de sortie sur du matériel Edge.Cette rigueur architecturale est la seule voie pour garantir une fluidité totale et éviter l'amnésie de session lors de générations complexes.

#### 5. Gestion de la Connaissance et Économie de Tokens

Le défi majeur d'un assistant local est la mémoire à long terme sans "amnésie catastrophique". Si le RAG classique ou le Graph RAG sont souvent mis en avant, la réalité de la production impose une vision plus tranchée :  **le futur du RAG n'est pas le GraphRAG** , trop complexe à maintenir et gourmand en contexte.Le système doit privilégier le  **Composite Retrieval**  :

1. **SQL pour filtrer :**  Utilisation d'une base relationnelle pour réduire drastiquement l'espace de recherche (ex: filtrer par date, projet ou catégorie).  
2. **Vecteurs pour rapprocher :**  Recherche sémantique uniquement sur le sous-ensemble filtré pour identifier la pertinence.Cette méthode "SQL pour filtrer, Vecteurs pour rapprocher" est cruciale pour l'économie de tokens. Les agents IA sont intrinsèquement  **récursifs**  (boucle de raisonnement) ; sans un filtrage strict, le "prompt stuffing" (bourrage de contexte) génère un effet boule de neige où une simple question consomme 60 000 tokens. Le Composite Retrieval garantit que seul le savoir strictement nécessaire est injecté dans le prompt, permettant l'usage de modèles locaux plus légers (8B à 14B paramètres) sans sacrifier l'intelligence sémantique.

#### 6. Écosystème Agentique et Interopérabilité des Outils

Pour devenir véritablement "agentique", Jarvis doit passer du modèle de langage pur à un acteur capable d'impacter son environnement via une boucle  **ReAct (Thought-Act-Observation)** .L'unification des capacités repose sur le  **Model Context Protocol (MCP)**  :

* **Serveurs MCP :**  Des micro-services indépendants gérant les accès (Base SQL, APIs domotiques, outils de recherche).  
* **Client MCP :**  Jarvis, qui orchestre ces outils en fonction de ses "pensées".L'utilisation d'un daemon comme  **OpenClaw**  permet l'exécution de code local pour des tâches complexes (manipulation de fichiers, calculs). L'écosystème intègre des outils de  **Deep Research local**  (scraping via Playwright/Jina) et un stockage de connaissances structuré en Markdown ( **LLM Wiki** ). Cette approche transforme Jarvis en un agent capable de mener des recherches approfondies de manière autonome avant de formuler une réponse, tout en sécurisant les accès par un contrôle strict des privilèges d'exécution.

#### 7. Accès Distant et Sécurisation du Système

Un assistant personnel local doit être accessible de n'importe où, sans pour autant devenir une faille de sécurité. L'ouverture de ports est proscrite. La solution réside dans **Tailscale (VPN mesh WireGuard)**. Il permet de créer un réseau privé virtuel chiffré de bout en bout, contournant le CGNAT et les IPs dynamiques sans exposition publique. Pour le contrôle d'interface graphique (GUI) à distance,  **RustDesk**  est l'outil de référence, offrant une expérience proche du "Cloud Gaming" pour coder ou naviguer depuis un mobile.*Note technique critique :*  Sur Ubuntu, il est impératif de désactiver Wayland au profit de  **Xorg**  pour permettre à RustDesk de capturer l'écran et d'injecter les événements d'entrée sans restriction de sécurité système. Cette infrastructure garantit un accès permanent, sécurisé et performant à l'ensemble du système Jarvis.

#### 8. Plan d'Action : Feuille de Route d'Implémentation

##### Phase 1 : Infrastructure Matérielle

Assemblage du cerveau (Jetson Orin Nano pour le support CUDA) et des satellites ESP32.

* **Conseil Pro :**  Utilisez exclusivement des adresses IP fixes Tailscale (100.x.y.z) pour la communication inter-nœuds afin d'éviter toute rupture de service liée au DHCP.

##### Phase 2 : Firmware & Perception

Codage du streaming audio PCM via UDP sur ESP32 et intégration de la détection TinyML.

* **Conseil Pro :**  Supprimez tout mécanisme de "handshake" réseau lourd pour le flux audio afin de garantir une latence zéro-jitter.

##### Phase 3 : Noyau IA & Optimisation

Compilation de llama.cpp et whisper.cpp avec support CUDA. Mise en place du KV Caching.

* **Conseil Pro :**  Vérifiez le support des instructions matérielles NEON/CUDA lors de la compilation pour maximiser le débit de tokens.

##### Phase 4 : Intelligence Sémantique

Déploiement du  **Composite Retrieval**  (PostgreSQL avec pgvector). Standardisation du savoir en Markdown.

* **Conseil Pro :**  Le nettoyage des données en amont (normalisation Markdown) est plus rentable que l'utilisation d'un modèle plus gros.

##### Phase 5 : Écosystème Agentique & Accès

Déploiement des serveurs MCP et configuration de Tailscale/RustDesk.

* **Conseil Pro :**  Désactivez Wayland pour  **Xorg**  avant le premier test distant et utilisez le  **Hacker’s Keyboard**  sur mobile pour disposer des touches Ctrl/Alt/Esc indispensables au développement. Cette architecture auto-hébergée offre une pérennité et une réactivité inaccessibles aux solutions cloud. En maîtrisant chaque couche, du kernel C++ au protocole VPN, vous forgez un système réellement privé, performant et capable d'évoluer au rythme des innovations de l'Edge AI.
