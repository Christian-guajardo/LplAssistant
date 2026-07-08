https://github.com/google-deepmind/deepmind-research/tree/master/wikigraphs
https://github.com/google-deepmind/deepmind-research/tree/master/continual_learning
https://github.com/google-deepmind/deepmind-research/blob/master/README.md
https://github.com/Alibaba-NLP/DeepResearch.git
https://github.com/dzhng/deep-research.git
https://github.com/langchain-ai/open_deep_research.git
https://github.com/langchain-ai/local-deep-researcher.git
https://github.com/u14app/deep-research.git
https://github.com/jina-ai/node-DeepResearch.git
https://github.com/assafelovic/gpt-researcher.git
https://github.com/Green-PT/honey-for-devs.git
https://github.com/ggml-org/.github/blob/master/profile/README.md
https://github.com/ggml-org/llama.cpp.git

Pour chaque url que je t'envoie, si c'est un site internet, fetch la page, si c'est un repo Github clone le dans un dossier de stockage ici ou tu va cloner les autres puis cherche les README.md et les fichiers de documentation pour extraire les informations pertinentes sur le projet.
Conseille, généralement dans les README.md et autres docs il peux y avoir des liens vers d'autres ressources, des points de recherches, du savoir qui permettrais d'aider au développement de ce projet de Deep Research. Ces information peuvent possiblement être très importantes pour le développement du projet. Et etant donné qu'il faut penser besoin avant solution, il me semble évidement que tenter de récupérer ces informations en utilisant notre outils permettras de voir nos limites actuelles et de nous aider à mieux comprendre les besoins du projet. et de trouver une solution adapté afin de pouvoir correctement extraire ces informations. Tu vois la boucle, tu trouve des liens ou des ressources d'intéret que l'on ne peux pas récupérer automatiquement avec notre outils, alors c'est un besoin et il faut le résoudre afin de pouvoir récupérer ces informations et les utiliser pour le développement du projet.

Méthode qui pourrait être utilisée pour extraire les informations pertinentes des README.md et autres fichiers de documentation plus tard :
On va extraire tout le jus de tout ces repo et autres liens; on va leur extraires tout leur savoir et connaissances et experiences pour les mettre à disposition dans notre projet de Deep Research. Cela inclut les méthodologies, les algorithmes, les datasets utilisés, les résultats expérimentaux, et toute autre information pertinente qui pourrait enrichir notre compréhension et notre capacité à développer notre solution innovante.

Là ou on a de la chance, j'ai déjà clone le repo d'un gars que je connais et qui est plutôt bon et qui avais fait un poc en python que j'ai déposer dans le dossier @star-hengxing_deep-research/ et que tu pourras bouger dans le dossier de stockage après une fois que tu auras terminé ton analyse des points pertinents à récupérer et de son fonctionnement et de son pipeline.

Et deuxième point, c'est que j'ai déjà également tenter un poc basé sur beaucoup de discussion et de recherche sur le sujet, et j'ai déposé ce poc dans le dossier @masterlaplace_deep-research/ et que tu pourras également bouger dans le dossier de stockage après une fois que tu auras terminé ton analyse des points pertinents à récupérer et de son fonctionnement et de son pipeline. Pense bien à lire tout les *.md parce que tout peut-être pertinent.

Comme j'imagine que après tout ce travail de recherche et d'extraction d'informations, on aurras beaucoup trop d'information, je te propose de procéder par étapes; c'est que pour chaque lien et repo tu créer un document uniformisé qui résume tout ce que tu à pu apprendre et ce qui est utile, bref; et ensuite on vas pouvoir compact tout ces rapport en un gigantesque rapport qui contiendras tout le savoir en un .md qui serviras à la fois de point de référence et de source de vérité et sera utiliser pour construire le plan d'implementation. On vas avoir une bonne idée de ce que l'on veux faire et comment le faire, je pense que l'on pourra ensuite se concentrer sur la création de notre propre outils d'agentique basé sur llama.cpp. Et pour ça, je pense qu'il faudra que l'on fasse un gros travail de recherche et d'analyse des différents outils d'agentique existants, pour voir ce qui fonctionne bien et ce qui ne fonctionne pas, et pour voir comment on peut s'en inspirer pour créer notre propre outil.

Pour ce qui est de l'implementation au global, on vas créer notre propre outils d'agentique basé sur llama.cpp
https://github.com/anomalyco/opencode.git
https://github.com/OpenRouterTeam/.github/blob/main/profile/README.md
Et je ne sais pas si tu vois comment fonctionne open claw et si c'est peut-être pertinent pour notre projet, mais je pense que ça pourrait être intéressant de regarder comment ils ont implémenté leur système d'agentique et voir si on peut s'en inspirer pour notre propre outil basé sur llama.cpp.

Aussi le plan original de ce projet est basé sur ce rapport @Rapport_d-Implémentation_Conception_d-un_Assistant_Personnel_Jarvis_Local_et_Haute_Performance.md; tu peux le lire rapidement pour comprendre ce que cherche à faire ce projet, et de même pour le @README.md qui est normalement à jour.

Notre plus gros avantages c'est que nous n'avons pas besoin d'interface visuel ou tout autres choses marketing, et pas besoin d'intégration dans vscode ou autres, c'est notre projet; les seules choses qui importent sont est ce que ça marche et est ce que c'est ultra optimisé (bare metal ?) on peux utilisé les stack technique que l'on veux et adapter au projet. de toute façon ça tourneras sur un serveur dédié et on pourra tout optimiser pour que ça tourne le plus efficacement possible. L'objectif est de créer un outil d'agentique qui soit capable de récupérer, analyser les onformations. Et pour atteindre ce but nous allons faire un gros travail de recherche et d'extraction d'informations pertinentes à partir des différents repos et ressources que nous avons listés.

Bonne chance pour le projet, et n'hésite pas à me demander de l'aide pour l'extraction d'informations ou pour toute autre question liée au développement de notre outil d'agentique basé sur llama.cpp.
