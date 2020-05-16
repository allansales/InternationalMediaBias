# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances
from pyemd import emd
import unidecode
import sys

PUNCTUATION = u'[^a-zA-Z0-9àáéíóúÀÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇäöüÄÖÜ]' # define news punctuation 
SENTENCE_SIZE_THRESHOLD = 2 # Minimum length of a text

path = sys.argv[1] # path to file
tex_lang = sys.argv[2] # language of articles: i.e. "eng","por","deu","spa","ita"
wiki_model = sys.argv[3] # wikipedia word embedding: i.e. '../wordvectors/data/pt.bin'
outfile = sys.argv[4] # output filename


### Definindo Lexicons
#  Lexicons definition
portuguese_dict = {
    "argument":"em_funcao afinal so ainda_que e e_nao como como_consequencia em_decorrencia como_se sequer quando_muito enfim pelo_menos ao_contrario porque por_causa a_par_disso alem mas mas_tambem por_sinal consequentemente contrariamente apesar_de senao portanto por_conseguinte pois para_tanto por_essa_razao contudo uma_vez_que no_entanto quanto se_acaso incluindo inclusive no_intuito em_lugar em_suma em_vez no_mesmo_sentido no_caso no_minimo por_sua_vez muito_menos logicamente entretanto mais_que tampouco nao_obstante nao_mais_que nem nem_mesmo nao_so somente pelo_contrario de_um_lado por_outro_lado ou ou_entao ao_menos do_contrario desde_que ja_que tao de_certa_forma tanto tanto_que de_modo_que isso_e fora_isso nessa_direcao entao a_unica por_isso assim na_medida_que a_ponto menos_que ate nao_fosse_isso quando enquanto que ainda se",
    "modalization":"aconselhar aparente basico acreditar puder claro conveniente decidir negar dificil duvida dever efetivo evidente exato esperar justo fundamental tinha tem ter crer tendo tive imaginar importante provavel limitar logico obrigatorio natural necessario precisar obvio parecer facultativo talvez possivel poder predominar presumir real recomendar certo procurar falar bastar fato supor seguro achar verdade inegavel indubitavel",
    "pressuposition":"reconhecer admitir ja esclarecer aguentar antes gabar continuar verificar iniciar compreender corrigir atual demonstrar detectar descobrir aturar desculpar sentir prever esquecer conseguir adivinhar escutar ignorar comecar interromper saber deixar ouvir olhar atentar perder momento constatar notar agora observar perceber recordar enxergar lembrar retirar reparar ver pressentir desde parar tolerar tratar entender desvendar revelar ainda",
    "sentiment":"abominar admirar afligir agitar alarmar surpreender enervar apaziguar apreciar ambicionar pasmar atazanar azucrinar recear menosprezar enfeiticar entediar incomodar alegrar prezar consolar confundir constranger contemplar contentar contrariar cobicar acovardar aporrinhar deslumbrar enganar encantar iludir depreciar deprimir desesperar desprezar detestar desapontar desestimular desencantar desiludir desinteressar desgostar desorientar descontentar distrair conturbar envergonhar encorajar curtir entreter entusiasmar estimar exaltar exasperar empolgar execrar enamorar fascinar temer perdoar espantar frustar apoquentar descontrolar grilar amolar alucinar hostilizar chocar odiar assombrar magoar idolatrar impressionar endividar indignar enfurecer inibir intimidar intrigar irar irritar lastimar decepcionar gostar amar malquerer maravilhar chatear lamentar obcecar estranhar ofender superestimar abalar louvar aquietar enraivecer sensibilizar tranquilizar rejeitar repelir arrepender recalcar repudiar respeitar transtornar revoltar entristecer escandalizar assustar seduzir sossegar comover abater acalmar melindrar estimular aticar sublimar reprimir simpatizar apavorar aterrorizar emocionar tolerar atormentar traumatizar subestimar inquietar reverenciar querer debilitar desejar preocupar adorar",
    "valoration":"absoluto quase muito algum aproximado melhor grande amplo comum completo consideravel constante valer categorico sempre mal exagero excessivo exclusivo expresso extremo cerca definitivo franco franqueza frequente pleno geral bem otimo feliz elevado enorme imenso incrivel menos leve pouco maioria mero minoria mais bom normal ocasional pena prazer pobre eventual praticamente preciso preferir principal particular bastante raro razoavel relativo rico rigor escasso significativo simples minimo pequeno tao tanto pesar especial estrito excelente alto demais total tremenda tipico lamentavel usual pouquissimo generalizado pior"
}

english_dict = {
    "argument":"according_to after_all alone although and and_not as as_a_consequence as_a_result as_if as_well at_best at_last at_least backwards because because_of besides_that beyond but but_also by_the_way consequently conversely despite downside either even_though for for_this_purpose for_this_reason fully hence however how_much in_case including inclusive in_order in_place in_short instead in_the_same_vein in_this_case in_this_way in_turn let_alone logically meanwhile more_than neither nevertheless no_more_than nor not_even not_only only on_the_contrary on_the_one_hand on_the_other_hand or or_else other_than_that otherwise provided since so somehow so_much so_much_so_that so_that that_is that_is_why that_way then the_only therefore thus to_the_extent_that to_the_point unless until were_it_not_for_that when while who yet whether",
    "modalization":"advise apparent basic believe can clear-cut convenient decide deny difficult doubt duty effective evident exact expect fair fundamental had has have have_faith_in having i_had imagine important likely limit logical mandatory natural necessary need obvious opinion optional perhaps possible power predominate presume real recommend right search speak suffice suit suppose sure think truth undeniable undoubted",
    "pressuposition":"acknowledge admit already arify bear beforehand brag carry_on check commence comprehend correct current demonstrate detect discover endure excuse feel foresee forget get guess hear ignore initiate interrupt knowledge let listen look look_out miss moment note notice now observe perceive recall regard remember remove repair see sense since stop tolerate treat understand unravel unveil yet",
    "sentiment":"abhor admire afflict agitate alarm amaze annoy appease appreciate aspire astound atazanar azucrinar be_afraid belittle bewitch bore bother celebrate cherish comfort confuse constrain contemplate content contradict covet coward crowd dazzle deceive delight delude depreciate depress despair despise detest disappoint discourage disenchant disillusion disinterest dislike disorientate displease distract disturb embarrass encourage enjoy entertain enthuse estimate exalt exasperate excite execute fall_in_love fascinate fear forgive frighten_away frustrate fuss get_out_of_hand grill grind hallucinate harass hatch hate haunt hurt idolize impress indebtedness indign infuriate inhibit intimidate intrigue irar irritate lament let_down like love malquerer marvel mind mourn obsess odd offend overestimate overwhelm praise quiet rage raise_awareness reassure reject repel repent repress repudiate respect revere revolt sadden scandalize scare seduce settle_down shake shoot_down soothe sore stimulate stir sublimate suppress sympathize terrify terrorize thrill tolerate torment traumatize underestimate upset venerate want weaken wish worry worship",
    "valoration":"absolute almost a_lot any approximate better big broad common complete considerable constant count emphatic ever evil exaggeration excessive exclusive expressed extreme fence final frank frankness frequent full general good great happy high huge immense incredible less light little majority mere minority more nice normal occasional pity pleasure poor possible practically precise prefer principal private quite_a_lot rare reasonable relative rich rigour scarce significant simple slightest small so so_much sorrow special strict superb tall too_much total tremendous typical unfortunate usual very_little widespread worst"
}

german_dict = {
    "argument":"nach immerhin allein obwohl und und_nicht wie infolgedessen als_folge_von als_ob wenigstens bestenfalls endlich jedenfalls rueckwaerts denn wegen ausserdem jenseits aber aber_auch uebrigens folglich umgekehrt trotz nachteil deswegen deshalb fuer zu_diesem_zweck aus_diesem_grund doch seit jedoch wie_viel fuer_den_fall einschliesslich inklusive in_der_reihenfolge an_ort_und_stelle kurz_gesagt stattdessen in_gleicher_weise in_diesem_fall mindestens wiederum geschweige_denn logischerweise in_der_zwischenzeit mehr_als weder trotzdem nicht_mehr_als auch_nicht nicht_einmal nicht_nur lediglich im_gegenteil einerseits andererseits entweder oder_doch zumindest anderweitig bereitgestellt da so irgendwie so_sehr so_sehr_dass so_dass das_heisst ansonsten auf_diese_art_und_weise dann der_einzige auf_grund_von somit insoweit_als auf_den_punkt_gebracht es_sei_denn bis ohne_das_waere_es_nicht_so wann waehrend wer dennoch ob",
	"modalization":"beraten offensichtlich grundlegend glauben dose klar_umrissen praktisch entscheiden leugnen schwierig zweifel pflicht effektiv offenkundig genau erwarten fair fundamental hatte hat haben ueberlegen habend ich_hatte sich_einbilden wichtig wahrscheinlich limit logisch obligatorisch natuerlich notwendig bedarf klar meinung optional vielleicht moeglich macht vorherrschen annehmen wirklich empfehlen rechts suche sprechen ausreichen klage vermuten sicher denken wahrheit unbestreitbar zweifellos",
	"pressuposition":"bestaetigen zugeben bereits klaeren baer vorher prahlen weitermachen pruefen anfangen verstehen richtig aktuell demonstrieren erkennen entdecken ertragen entschuldigung gefuehl voraussehen vergessen bekommen erraten zuhoeren ignorieren beginnen unterbrechen wissen lassen anhoeren aussehen aufpassen verpassen moment hinweis merken jetzt beobachten wahrnehmen rueckruf sehen sich_erinnern entfernen reparatur ansehen sinn da haltestelle tolerieren leckerbissen denken entwirren enthuellen dennoch",
	"sentiment":"verabscheuen bewundern belasten sich_bewegen alarm ueberraschen argern besaenftigen schaetzen streben verblueffen atazanar azukrinar angst_haben herabsetzen verzaubern bohrung aufregen jubeln hoch_achten jammern verwirren einschraenken nachdenken inhalt widersprechen begehren feigling menge blendend taeuschen freude truegen abschreiben niederdruecken verzweiflung verachten hassen enttaeuschung abschrecken desillusionieren herunterlassen desinteresse nicht_moegen desorientieren missfallen ablenken stoeren in_verlegenheit_bringen ermutigen geniessen unterhalten sich_begeistern schaetzung verherrlichen erbosen anregen ausfuehren sich_verlieben faszinieren angst verzeihen verscheuchen frustrieren veraergert ausser_kontrolle_geraten grill schleifen halluzinieren belaestigen klappe hass lieblingsort verletzt vergoettern beeindrucken verschuldung empoeren wuetend_machen hemmen einschuechtern intrige irrar reizen klage enttaeuschen wie liebe verleumder wunder nerven trauern besessen merkwuerdig beleidigen ueberbewerten schuetteln lob ruhig wut bewusstseinsbildung versichern ablehnen abwehren bereuen unterdruecken verwerfen respekt durcheinanderbringen revolte traurig_machen einen_skandal_hervorrufen aengstigen verfuehren sich_beruhigen ruehren abschiessen beruhigen wund ermuntern schueren sublimieren zurueckhalten mitfuehlen angst_einjagen terrorisieren nervenkitzel tolerieren quaelerei traumatisieren unterschaetzen beunruhigen ehren wollen abschwaechen wunsch sorge anbetung",
	"valoration":"absolut fast oft irgendeine ungefaehr besser gross breit allgemein vollstaendig betraechtlich konstant anzahl nachdruecklich jemals boese uebertreibung uebertrieben exklusiv ausgedrueckt extrem zaun endspiel offen offenheit haeufig voll generell gut grossartig gluecklich hoch riesig immens unglaublich weniger licht wenig mehrheit nur minderheit mehr schon normal gelegentlich mitleid vergnuegen schlecht moeglich praktisch praezise bevorzugen auftraggeber privat ziemlich_viel selten vernuenftig relativ reich strenge knapp signifikant einfach geringste klein so so_sehr trauer speziell streng superb hochgewachsen zu_viel ganze enorm typisch ungluecklich ueblich sehr_wenig weit_verbreitet am_schlimmsten"
}

spanish_dict = {
    "argument":"en_base_a al_fin_y_al_cabo solo aunque e y_no a_medida_que como_consecuencia como_resultado como_si empatado en_el_mejor_de_los_casos por_fin como_minimo al_reves porque a_causa_de aparte_de_eso mas_alla sin_embargo sino_tambien por_cierto por_consiguiente a_la_inversa a_pesar_de aspecto_negativo por_lo_tanto en_consecuencia a_favor_de con_este_proposito por_este_motivo con_todo puesto_que no_obstante cuanto por_si con_detalle_de integrador en_orden en_su_sitio en_resumen en_cambio del_mismo_modo en_este_caso por_lo_menos uno_por_uno por_no_hablar_de logicamente entretanto mas_de ninguno_de_los_dos con_todo_y_eso no_mas_de ni ni_siquiera no_solo solamente por_el_contrario por_un_lado por_otro_lado o entonces  por_lo_menos de_lo_contrario suministrado ya_que por_lo_que de_alguna_manera tanto tanto_es_asi_que para_que es_decir ademas_de_eso de_esa_forma luego el_unico por_eso asi en_la_medida_en_que hasta_el_punto salvo_que hasta si_no_fuera_por_eso cuando si_bien quien todavia usted_mismo",
	"modalization":"aconsejar obvio elemental creer bote claro comodo decidir renegar dificultoso dudar aranceles efectivo incuestionable exactos esperar bueno primordial tenia tiene tener comprender que_tiene Yo_tenia imaginar significativo probable tope logicos imperativo innato imprescindible menester obvios postura optativo a_lo_mejor factible de_energia predominar presumir de_verdad sugerir reparar busqueda hablar bastar adecuarse suponer inequivoco pensar verdad incontestable indudable",
	"pressuposition":"reconocer dejar_entrar de_por_si esclarecer dar_a_luz de_antemano fanfarronear seguir_adelante chequeo comenzar concebir correctos corriente manifestar notar darse_cuenta_de aguantar disculpar estar prever olvidar conseguir conjetura prestar_atencion pasar_por_alto empezar interrumpir conocimiento dejar oir aspecto estar_atento perder ocasion apuntar letrero ahora guardar entender retiro sede acordarse quitar remediar ver sentir puesto_que detenerse tolerar obsequiar imaginar desentranar descubrir sin_embargo",
	"sentiment":"aborrecer admirar afligir conmocion despertador asombrar fastidiar apaciguar agradecer aspirar atraicoar atazanar azucrinar tener_miedo restar hechizar aburrir preocuparse ovacion apreciar consuelo desconcertar constrenir pensar contento contradecir codiciar cobarde apinamiento resplandor enganar delicia pasmarse depreciarse reducir desesperacion despreciar detestar defraudar desanimar desenganar desilusionar desinteres repugnancia desorientar descontento distraer desordenar avergonzar animar gozar recibir entusiasmar valoracion exaltar exasperar suscitar cumplir enamorarse fascinar temerse perdonar ahuyentar frustrar disgustado irse_de_las_manos parrilla moler alucinar acosar escotilla odio guarida apesadumbrado idolatrar impronta endeudamiento indignar enfurecer impedir amedrentar conspirar iris irritar lamentar decepcionar parecido_a amor malquerer prodigio molestar estar_de_luto obsesionar singular cometer_un_crimen sobrevaloracion batir elogio sosegado rabia sensibilizar tranquilizar marginado repelerse arrepentirse reprimir negar respetar trastornar sublevarse entristecer escandalizar asustar seducir calmarse conmover derribar calmar adolorido estimular atizar sublimar trastocar simpatizar aterrorizar terrificar estremecimiento aguantar tormento traumatizar subestimar inquietar adoracion deseo socavar anhelar preocupar encanta",
	"valoration":"categorico por_poco mucho algo_de aproximado mejor enorme tia corriente integro apreciable ininterrumpido computo cabal jamas malvado engrandecimiento desmesurado privativo expresado extremas vallado colofon honesto franqueza frecuentar completo generalizado bueno grandioso afortunado elevado gigantesco vasto increibles menos liviano pequeno mayoritario simple minoritario mas bien habitual esporadico lastima delicia deficiente factible practicamente exactos preferir principal reservado bastante infrecuente asequible pariente acaudalado severidad escaso significativo sencillo minimo banal por_lo_que tanto angustia especial riguroso soberbio altas demasiado total inmenso tipico desdichado usual muy_poco difundida mas_grave"
}

italian_dict = {
    "argument":"in_base_a del_resto singolarmente sebbene e e_non in_qualita_di di_conseguenza percio come_se regolare al_massimo infine come_minimo all'indietro perche a_causa_di a_parte_questo oltre ma ma_anche comunque conseguentemente invece malgrado rovescio_della_medaglia pertanto dunque visto_che a_questo_scopo per_questo_motivo eppure poiche in_qualunque_modo quanto nel_caso compreso comprensivo nell'ordine in_atto in_breve per_contro nella_stessa_ottica in_questo_caso per_lo_meno a_sua_volta figuriamoci logicamente nel_frattempo piu_di nessuno_dei_due ciononostante non_piu_di ne nemmeno non_solo solo_che al_contrario da_un_lato d'altra_parte o ovvero almeno altrimenti a_condizione_che dato_che di_modo_che in_qualche_modo tanto tanto_che cosicche cioe a_parte_cio tal_modo poi l'unica quindi cosi nella_misura_in_cui al_punto a_meno_che fino_a se_non_fosse_stato_per_questo quando lasso_di_tempo che ancora voi_stessi",
	"modalization":"consigliare apparente elementare credere essere_in_grado chiara agevole decidere negare impegnativa dubbio mansione effettivo ovvia puntuale aspettarsi leale fondamentale aveva ha tenere pensare avere Ho_avuto immaginarsi importante verosimilmente soglia logico inderogabile naturalistico necessario necessitare scontate avviso facoltative magari fattibile potere prevalere presumere vero raccomandare destro ricerca parlare basti andare_bene supporre certo riflettere veridicita incontrovertibile indiscussa",
	"pressuposition":"riconoscere ammettere gia chiarire orso in_anticipo vantarsi continuare assegno cominciare comprendere esatta vigente dimostrare individuare scoprire durare pretesto sensazione prevedere dimenticarsi diventare congettura ascoltare ignorare iniziare interrompere consapevolezza lasciare ascolto occhiata guardare_fuori insuccesso attimo banconota avviso adesso seguire avvertire richiamo vedere ricordare allontanare riparazione osservare accezione poiche arrestarsi tollerare chicca capire dipanarsi svelare eppure",
	"sentiment":"aborto ammirare afflitto agitazione sveglia stupire seccatura calmarsi rivalutarsi aspirare sorprendere atazanar azucrinaro avere_paura sminuire strega foro inconveniente tifo preziose conforto confondere costrizione contemplare tenore smentire brama vigliacco schiera abbagliamento imbrogliare godimento illusione deprezzarsi premere sconforto disprezzo detest amareggiare scoraggiare disincanto ostacolare disinteresse avversione disorientare malcontento distrarsi disturbo imbarazzo incentivare divertirsi divertire entusiasmante stimare esaltare esasperata eccitarsi espletare innamorarsi affascinare timore perdonare spaventare_via frustrato turbata sfuggire_di_mano griglia routine allucinato molestare portello odio ritrovo ferita idolatrare impressionare indebitamento indignarsi infuriare inibire intimidire intrigo ferro irritazione lamento deludere tipo affetto malquerer prodigio noia piangere ossessione bizzarro offendere sopravvalutazione scossa elogio quieta imperversare sensibilizzare_l'opinione_pubblica rassicurare scarto respingere pentirsi reprimere rinnegare stima turbare insurrezione triste scandalizzare spaventare sedurre stabilirsi commuovere abbattere lenire indolenzito promuovere scalpore sublimato opprimere simpatizzare terrorizzare intimorire ebbrezza tollerare supplizio traumatizzare sottostima sconvolto venerazione volere indebolirsi desiderio preoccupazione piacere",
	"valoration":"imprescindibile quasi molto qualcuno approssimato migliore vasto ampia diffuso intero cospicuo costante computo enfatico mai malvagita esagerazione eccessive esclusiva formulata estreme staccionata definitivo sinceri franchezza assiduo completo generale valido fantastica contenta elevati gigantesca immane incredibile di_meno leggero piccolo maggioranza mero minorita di_piu buono normale saltuaria commiserazione godimento povera fattibile praticamente esatto prediligere mandante privato un_bel_po'. insolito sensata parente facoltoso severita scarsa significativo semplice minima modesto di_modo_che tanto cordoglio particolare severi superlativo alto troppo totale tremenda tipico deplorevole solito molto_poco capillare peggio"
}

# Mapping words in lexicons
map_lexicons_por = {" em funcao ":" em_funcao ", " ainda que ":" ainda_que ", " e nao ":" e_nao ", " como consequencia ":" como_consequencia ", " em decorrencia ":" em_decorrencia ", " como se ":" como_se ", " quando muito ":" quando_muito ", " pelo menos ":" pelo_menos ", " ao contrario ":" ao_contrario ", " por causa ":" por_causa ", " a par disso ":" a_par_disso ", " mas tambem ":" mas_tambem ", " por sinal ":" por_sinal ", " apesar de ":" apesar_de ", " por conseguinte ":" por_conseguinte ", " para tanto ":" para_tanto ", " por essa razao ":" por_essa_razao ", " uma vez que ":" uma_vez_que ", " no entanto ":" no_entanto ", " se acaso ":" se_acaso ", " no intuito ":" no_intuito ", " em lugar ":" em_lugar ", " em suma ":" em_suma ", " em vez ":" em_vez ", " no mesmo sentido ":" no_mesmo_sentido ", " no caso ":" no_caso ", " no minimo ":" no_minimo ", " por sua vez ":" por_sua_vez ", " muito menos ":" muito_menos ", " mais que ":" mais_que ", " nao obstante ":" nao_obstante ", " nao mais que ":" nao_mais_que ", " nem mesmo ":" nem_mesmo ", " nao so ":" nao_so ", " pelo contrario ":" pelo_contrario ", " de um lado ":" de_um_lado ", " por outro lado ":" por_outro_lado ", " ou entao ":" ou_entao ", " ao menos ":" ao_menos ", " do contrario ":" do_contrario ", " desde que ":" desde_que ", " ja que ":" ja_que ", " de certa forma ":" de_certa_forma ", " tanto que ":" tanto_que ", " de modo que ":" de_modo_que ", " isso e ":" isso_e ", " fora isso ":" fora_isso ", " nessa direcao ":" nessa_direcao ", " a unica ":" a_unica ", " por isso ":" por_isso ", " na medida que ":" na_medida_que ", " a ponto ":" a_ponto ", " menos que ":" menos_que ", " nao fosse isso ":" nao_fosse_isso "}
map_lexicons_eng = {" according to ":" according_to ", " after all ":" after_all ", " and not ":" and_not ", " as a consequence ":" as_a_consequence ", " as a result ":" as_a_result ", " as if ":" as_if ", " as well ":" as_well ", " at best ":" at_best ", " at last ":" at_last ", " at least ":" at_least ", " because of ":" because_of ", " besides that ":" besides_that ", " but also ":" but_also ", " by the way ":" by_the_way ", " even though ":" even_though ", " for this purpose ":" for_this_purpose ", " for this reason ":" for_this_reason ", " how much ":" how_much ", " in case ":" in_case ", " in order ":" in_order ", " in place ":" in_place ", " in short ":" in_short ", " in the same vein ":" in_the_same_vein ", " in this case ":" in_this_case ", " in this way ":" in_this_way ", " in turn ":" in_turn ", " let alone ":" let_alone ", " more than ":" more_than ", " no more than ":" no_more_than ", " not even ":" not_even ", " not only ":" not_only ", " on the contrary ":" on_the_contrary ", " on the one hand ":" on_the_one_hand ", " on the other hand ":" on_the_other_hand ", " or else ":" or_else ", " other than that ":" other_than_that ", " so much ":" so_much ", " so much so that ":" so_much_so_that ", " so that ":" so_that ", " that is ":" that_is ", " that is why ":" that_is_why ", " that way ":" that_way ", " the only ":" the_only ", " to the extent that ":" to_the_extent_that ", " to the point ":" to_the_point ", " were it not for that ":" were_it_not_for_that ", " have faith in ":" have_faith_in ", " i had ":" i_had ", " carry on ":" carry_on ", " look out ":" look_out ", " be afraid ":" be_afraid ", " fall in love ":" fall_in_love ", " frighten away ":" frighten_away ", " get out of hand ":" get_out_of_hand ", " let down ":" let_down ", " raise awareness ":" raise_awareness ", " settle down ":" settle_down ", " shoot down ":" shoot_down ", " a lot ":" a_lot ", " quite a lot ":" quite_a_lot ", " so much ":" so_much ", " too much ":" too_much ", " very little ":" very_little "}
map_lexicons_ger = {" und nicht ":" und_nicht ", " als folge von ":" als_folge_von ", " als ob ":" als_ob ", " aber auch ":" aber_auch ", " zu diesem zweck ":" zu_diesem_zweck ", " aus diesem grund ":" aus_diesem_grund ", " wie viel ":" wie_viel ", " fuer den fall ":" fuer_den_fall ", " in der reihenfolge ":" in_der_reihenfolge ", " an ort und stelle ":" an_ort_und_stelle ", " kurz gesagt ":" kurz_gesagt ", " in gleicher weise ":" in_gleicher_weise ", " in diesem fall ":" in_diesem_fall ", " geschweige denn ":" geschweige_denn ", " in der zwischenzeit ":" in_der_zwischenzeit ", " mehr als ":" mehr_als ", " nicht mehr als ":" nicht_mehr_als ", " auch nicht ":" auch_nicht ", " nicht einmal ":" nicht_einmal ", " nicht nur ":" nicht_nur ", " im gegenteil ":" im_gegenteil ", " oder doch ":" oder_doch ", " so sehr ":" so_sehr ", " so sehr dass ":" so_sehr_dass ", " so dass ":" so_dass ", " das heisst ":" das_heisst ", " auf diese art und weise ":" auf_diese_art_und_weise ", " der einzige ":" der_einzige ", " auf grund von ":" auf_grund_von ", " insoweit als ":" insoweit_als ", " auf den punkt gebracht ":" auf_den_punkt_gebracht ", " es sei denn ":" es_sei_denn ", " ohne das waere es nicht so ":" ohne_das_waere_es_nicht_so ", " klar umrissen ":" klar_umrissen ", " ich hatte ":" ich_hatte ", " sich einbilden ":" sich_einbilden ", " sich erinnern ":" sich_erinnern ", " sich bewegen ":" sich_bewegen ", " angst haben ":" angst_haben ", " hoch achten ":" hoch_achten ", " nicht moegen ":" nicht_moegen ", " in verlegenheit bringen ":" in_verlegenheit_bringen ", " sich begeistern ":" sich_begeistern ", " sich verlieben ":" sich_verlieben ", " ausser kontrolle geraten ":" ausser_kontrolle_geraten ", " wuetend machen ":" wuetend_machen ", " traurig machen ":" traurig_machen ", " einen skandal hervorrufen ":" einen_skandal_hervorrufen ", " sich beruhigen ":" sich_beruhigen ", " angst einjagen ":" angst_einjagen ", " ziemlich viel ":" ziemlich_viel ", " so sehr ":" so_sehr ", " zu viel ":" zu_viel ", " sehr wenig ":" sehr_wenig ", " weit verbreitet ":" weit_verbreitet ", " am schlimmsten ":" am_schlimmsten "}
map_lexicons_ita = {" in base a ":" in_base_a ", " del resto ":" del_resto ", " e non ":" e_non ", " in qualita di ":" in_qualita_di ", " di conseguenza ":" di_conseguenza ", " come se ":" come_se ", " al massimo ":" al_massimo ", " come minimo ":" come_minimo ", " a causa di ":" a_causa_di ", " a parte questo ":" a_parte_questo ", " ma anche ":" ma_anche ", " rovescio della medaglia ":" rovescio_della_medaglia ", " visto che ":" visto_che ", " a questo scopo ":" a_questo_scopo ", " per questo motivo ":" per_questo_motivo ", " in qualunque modo ":" in_qualunque_modo ", " nel caso ":" nel_caso ", " in atto ":" in_atto ", " in breve ":" in_breve ", " per contro ":" per_contro ", " nella stessa ottica ":" nella_stessa_ottica ", " in questo caso ":" in_questo_caso ", " per lo meno ":" per_lo_meno ", " a sua volta ":" a_sua_volta ", " nel frattempo ":" nel_frattempo ", " piu di ":" piu_di ", " nessuno dei due ":" nessuno_dei_due ", " non piu di ":" non_piu_di ", " non solo ":" non_solo ", " solo che ":" solo_che ", " al contrario ":" al_contrario ", " da un lato ":" da_un_lato ", " d'altra parte ":" d'altra_parte ", " a condizione che ":" a_condizione_che ", " dato che ":" dato_che ", " di modo che ":" di_modo_che ", " in qualche modo ":" in_qualche_modo ", " tanto che ":" tanto_che ", " a parte cio ":" a_parte_cio ", " tal modo ":" tal_modo ", " nella misura in cui ":" nella_misura_in_cui ", " al punto ":" al_punto ", " a meno che ":" a_meno_che ", " fino a ":" fino_a ", " se non fosse stato per questo ":" se_non_fosse_stato_per_questo ", " lasso di tempo ":" lasso_di_tempo ", " voi stessi ":" voi_stessi ", " essere in grado ":" essere_in_grado ", " Ho avuto ":" Ho_avuto ", " andare bene ":" andare_bene ", " in anticipo ":" in_anticipo ", " guardare fuori ":" guardare_fuori ", " avere paura ":" avere_paura ", " spaventare via ":" spaventare_via ", " sfuggire di mano ":" sfuggire_di_mano ", " sensibilizzare l'opinione pubblica ":" sensibilizzare_l'opinione_pubblica ", " di meno ":" di_meno ", " di piu ":" di_piu ", " un bel po'. ":" un_bel_po'. ", " di modo che ":" di_modo_che ", " molto poco ":" molto_poco "}
map_lexicons_spa = {" en base a ":" en_base_a ", " al fin y al cabo ":" al_fin_y_al_cabo ", " y no ":" y_no ", " a medida que ":" a_medida_que ", " como consecuencia ":" como_consecuencia ", " como resultado ":" como_resultado ", " como si ":" como_si ", " en el mejor de los casos ":" en_el_mejor_de_los_casos ", " por fin ":" por_fin ", " como minimo ":" como_minimo ", " al reves ":" al_reves ", " a causa de ":" a_causa_de ", " aparte de eso ":" aparte_de_eso ", " mas alla ":" mas_alla ", " sin embargo ":" sin_embargo ", " sino tambien ":" sino_tambien ", " por cierto ":" por_cierto ", " por consiguiente ":" por_consiguiente ", " a la inversa ":" a_la_inversa ", " a pesar de ":" a_pesar_de ", " aspecto negativo ":" aspecto_negativo ", " por lo tanto ":" por_lo_tanto ", " en consecuencia ":" en_consecuencia ", " a favor de ":" a_favor_de ", " con este proposito ":" con_este_proposito ", " por este motivo ":" por_este_motivo ", " con todo ":" con_todo ", " puesto que ":" puesto_que ", " no obstante ":" no_obstante ", " por si ":" por_si ", " con detalle de ":" con_detalle_de ", " en orden ":" en_orden ", " en su sitio ":" en_su_sitio ", " en resumen ":" en_resumen ", " en cambio ":" en_cambio ", " del mismo modo ":" del_mismo_modo ", " en este caso ":" en_este_caso ", " por lo menos ":" por_lo_menos ", " uno por uno ":" uno_por_uno ", " por no hablar de ":" por_no_hablar_de ", " mas de ":" mas_de ", " ninguno de los dos ":" ninguno_de_los_dos ", " con todo y eso ":" con_todo_y_eso ", " no mas de ":" no_mas_de ", " ni siquiera ":" ni_siquiera ", " no solo ":" no_solo ", " por el contrario ":" por_el_contrario ", " por un lado ":" por_un_lado ", " por otro lado ":" por_otro_lado ", "  por lo menos ":"  por_lo_menos ", " de lo contrario ":" de_lo_contrario ", " ya que ":" ya_que ", " por lo que ":" por_lo_que ", " de alguna manera ":" de_alguna_manera ", " tanto es asi que ":" tanto_es_asi_que ", " para que ":" para_que ", " es decir ":" es_decir ", " ademas de eso ":" ademas_de_eso ", " de esa forma ":" de_esa_forma ", " el unico ":" el_unico ", " por eso ":" por_eso ", " en la medida en que ":" en_la_medida_en_que ", " hasta el punto ":" hasta_el_punto ", " salvo que ":" salvo_que ", " si no fuera por eso ":" si_no_fuera_por_eso ", " si bien ":" si_bien ", " usted mismo ":" usted_mismo ", " que tiene ":" que_tiene ", " Yo tenia ":" Yo_tenia ", " a lo mejor ":" a_lo_mejor ", " de energia ":" de_energia ", " de verdad ":" de_verdad ", " dejar entrar ":" dejar_entrar ", " de por si ":" de_por_si ", " dar a luz ":" dar_a_luz ", " de antemano ":" de_antemano ", " seguir adelante ":" seguir_adelante ", " darse cuenta de ":" darse_cuenta_de ", " prestar atencion ":" prestar_atencion ", " pasar por alto ":" pasar_por_alto ", " estar atento ":" estar_atento ", " puesto que ":" puesto_que ", " sin embargo ":" sin_embargo ", " tener miedo ":" tener_miedo ", " irse de las manos ":" irse_de_las_manos ", " parecido a ":" parecido_a ", " estar de luto ":" estar_de_luto ", " cometer un crimen ":" cometer_un_crimen ", " por poco ":" por_poco ", " algo de ":" algo_de ", " por lo que ":" por_lo_que ", " muy poco ":" muy_poco ", " mas grave ":" mas_grave "}

def set_embeddings(wv_lang):
    wv_lang.init_sims()
    vocab_dict ={word.encode('utf-8'):vocab.index for word, vocab in wv_lang.vocab.items()}
    W = np.double(wv_lang.vectors_norm)
    return(vocab_dict, W)

### Carregando Word Embeddings
wv = KeyedVectors.load_word2vec_format(wiki_model, binary=True)
vocab_dict, W = set_embeddings(wv)

### Definindo Funções
# Convert word from text into lexicons
def word2lexicon(text, map_lexicons):
    text = text.decode('utf-8')
    text = re.sub(PUNCTUATION, " ", text).lower() # remove punctuation from text
    text = unidecode.unidecode(text) # remove accents
    for k, v in map_lexicons.items():
        text = text.replace(k,v)
    return text

# define settings of languages
def set_parameters(tex_lang):
    
    if tex_lang == "eng":
        lang = "english"
        lang_dict = english_dict
        map_lexicons = map_lexicons_eng
    
    if tex_lang == "por":
        lang = "portuguese"
        lang_dict = portuguese_dict
        map_lexicons = map_lexicons_por
        
    if tex_lang == "deu":
        lang = "german"
        lang_dict = german_dict
        map_lexicons = map_lexicons_ger

    if tex_lang == "spa":
        lang = "spanish"
        lang_dict = spanish_dict
        map_lexicons = map_lexicons_spa

    if tex_lang == "ita":
        lang = "italian"
        lang_dict = italian_dict
        map_lexicons = map_lexicons_ita

        
    return(lang, lang_dict, map_lexicons)

# function for processing text
def processSentences(text, lang):
    stop_words = stopwords.words(lang) # load stop words
    text = text.split() # split sentences by words
    text = [word for word in text if word not in stop_words] # Remove stopwords
    return " ".join(text)

# Check if the word is in the vocabulary
def check_value(word, vocab_dict):
    return (vocab_dict[word] if(word in vocab_dict) else 0)

# Compute the euclidean distances between the lexicons and the text
def lexicon_rate(lexicon, text, W, vocab_dict):
    vect = CountVectorizer(token_pattern="(?u)\\b[\\w-]+\\b", strip_accents=None).fit([lexicon, text])
    v_1, v_2 = vect.transform([lexicon, text])
    v_1 = v_1.toarray().ravel()
    v_2 = v_2.toarray().ravel()
    W_ = W[[check_value(w, vocab_dict) for w in vect.get_feature_names()]]
    D_ = euclidean_distances(W_)
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()
    D_ = D_.astype(np.double)
    D_ /= D_.max()
    lex=emd(v_1, v_2, D_)
    return(lex)

# Compute bias for each lexicon dimension
def wmd_ratings(text, lang_dict, W, vocab_dict):
    if(is_valid_text(text)):
        arg = lexicon_rate(lang_dict["argument"], text, W, vocab_dict)
        mod = lexicon_rate(lang_dict["modalization"], text, W, vocab_dict)
        val = lexicon_rate(lang_dict["valoration"], text, W, vocab_dict)
        sen = lexicon_rate(lang_dict["sentiment"], text, W, vocab_dict)
        pre = lexicon_rate(lang_dict["pressuposition"], text, W, vocab_dict)
        return arg, sen, val, mod, pre
    else :
        return -1, -1, -1, -1, -1

#news = pd.read_csv(path, index_col=None, sep=",")
news = pd.read_csv(path,sep="\t")
#news = news.head(30)

n, c = news.shape

sub = pd.DataFrame(columns=["arg", "sen", "val", "mod", "pre"], index = range(0,n))

import time
start_time = time.time()

# Set language configuration
lang, lang_dict, map_lexicons = set_parameters(tex_lang)

for index, article in news.iterrows():
    
    # Processing text
    text = word2lexicon(article["body"], map_lexicons)
    text = processSentences(text, lang)
    
    # Compute news bias
    arg, sen, val, mod, pre = wmd_ratings(text, lang_dict, W, vocab_dict)
    sub.loc[index,] = [arg, sen, val, mod, pre]
    
    if(index%1000==0):
        sub.to_csv(outfile+"_temp_"+tex_lang+".csv", index_label=False)
    
    #print('Index: {0} - Progress: {1:.2f} %'.format(index, float(index) / n*100 ), end='\r')
print("--- %s seconds ---" % (time.time() - start_time))

pd.concat([news, sub],axis=1).to_csv(outfile+"_"+tex_lang+".csv", index=False)
