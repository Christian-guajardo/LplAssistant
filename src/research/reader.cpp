#include "reader.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace laplace::research {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

// Supprime tout bloc <tag ...>...</tag> (insensible à la casse, non imbriqué
// pour les tags de structure — suffisant en pratique pour nav/footer/aside).
void strip_block(std::string& html, const std::string& tag) {
    const std::string lo = lower(html); // recalculé à chaque tag : tailles modestes
    std::string out;
    out.reserve(html.size());
    size_t pos = 0;
    const std::string open = "<" + tag;
    const std::string close = "</" + tag;
    while (pos < html.size()) {
        size_t b = lo.find(open, pos);
        // Vérifie que le match est bien un début de tag (<navX> ne compte pas).
        while (b != std::string::npos) {
            char next = b + open.size() < lo.size() ? lo[b + open.size()] : ' ';
            if (next == '>' || next == ' ' || next == '\t' || next == '\n' || next == '/')
                break;
            b = lo.find(open, b + 1);
        }
        if (b == std::string::npos) {
            out.append(html, pos, html.size() - pos);
            break;
        }
        out.append(html, pos, b - pos);
        size_t e = lo.find(close, b);
        if (e == std::string::npos) break; // tag jamais fermé : on coupe là
        e = lo.find('>', e);
        if (e == std::string::npos) break;
        pos = e + 1;
    }
    html = std::move(out);
}

void decode_entities(std::string& s) {
    struct Ent { const char* name; const char* repl; };
    static const Ent ents[] = {
        {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""},
        {"&#39;", "'"}, {"&apos;", "'"}, {"&nbsp;", " "}, {"&hellip;", "…"},
        {"&mdash;", "—"}, {"&ndash;", "–"}, {"&rsquo;", "'"}, {"&lsquo;", "'"},
        {"&ldquo;", "“"}, {"&rdquo;", "”"}, {"&eacute;", "é"}, {"&egrave;", "è"},
        {"&agrave;", "à"}, {"&ccedil;", "ç"}, {"&ecirc;", "ê"}, {"&ucirc;", "û"},
    };
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '&') {
            bool done = false;
            // Entité numérique &#123; / &#x1F;
            if (i + 2 < s.size() && s[i + 1] == '#') {
                size_t end = s.find(';', i);
                if (end != std::string::npos && end - i <= 10) {
                    long code = (s[i + 2] == 'x' || s[i + 2] == 'X')
                                    ? std::strtol(s.c_str() + i + 3, nullptr, 16)
                                    : std::strtol(s.c_str() + i + 2, nullptr, 10);
                    // Encodage UTF-8 minimal
                    if (code > 0 && code < 0x110000) {
                        if (code < 0x80) out += (char)code;
                        else if (code < 0x800) {
                            out += (char)(0xC0 | (code >> 6));
                            out += (char)(0x80 | (code & 0x3F));
                        } else if (code < 0x10000) {
                            out += (char)(0xE0 | (code >> 12));
                            out += (char)(0x80 | ((code >> 6) & 0x3F));
                            out += (char)(0x80 | (code & 0x3F));
                        } else {
                            out += (char)(0xF0 | (code >> 18));
                            out += (char)(0x80 | ((code >> 12) & 0x3F));
                            out += (char)(0x80 | ((code >> 6) & 0x3F));
                            out += (char)(0x80 | (code & 0x3F));
                        }
                        i = end + 1;
                        done = true;
                    }
                }
            }
            if (!done) {
                for (const auto& e : ents) {
                    size_t n = std::char_traits<char>::length(e.name);
                    if (s.compare(i, n, e.name) == 0) {
                        out += e.repl;
                        i += n;
                        done = true;
                        break;
                    }
                }
            }
            if (done) continue;
        }
        out += s[i++];
    }
    s = std::move(out);
}

} // namespace

bool looks_like_html(const std::string& body) {
    const std::string head = lower(body.substr(0, 512));
    return head.find("<!doctype html") != std::string::npos ||
           head.find("<html") != std::string::npos ||
           head.find("<head") != std::string::npos ||
           head.find("<body") != std::string::npos;
}

PageText html_to_text(const std::string& html_in) {
    PageText page;
    std::string html = html_in;

    // Titre avant tout nettoyage.
    {
        const std::string lo = lower(html);
        size_t b = lo.find("<title");
        if (b != std::string::npos) {
            b = lo.find('>', b);
            size_t e = lo.find("</title>", b);
            if (b != std::string::npos && e != std::string::npos) {
                page.title = html.substr(b + 1, e - b - 1);
                decode_entities(page.title);
                // compacte les blancs du titre
                std::string t;
                bool sp = false;
                for (char c : page.title) {
                    if (std::isspace((unsigned char)c)) { sp = true; continue; }
                    if (sp && !t.empty()) t += ' ';
                    sp = false;
                    t += c;
                }
                page.title = t;
            }
        }
    }

    // Commentaires HTML.
    for (size_t b; (b = html.find("<!--")) != std::string::npos;) {
        size_t e = html.find("-->", b);
        if (e == std::string::npos) { html.resize(b); break; }
        html.erase(b, e - b + 3);
    }
    // Blocs sans valeur informative.
    for (const char* tag : {"script", "style", "noscript", "template", "svg",
                            "nav", "footer", "header", "aside", "form", "iframe"})
        strip_block(html, tag);

    // Tags de bloc -> retours à la ligne ; <li> -> puce ; le reste est retiré.
    std::string text;
    text.reserve(html.size() / 2);
    const std::string lo = lower(html);
    for (size_t i = 0; i < html.size();) {
        if (html[i] == '<') {
            size_t e = html.find('>', i);
            if (e == std::string::npos) break;
            static const char* blocks[] = {"p", "div", "br", "tr", "h1", "h2", "h3",
                                           "h4", "h5", "h6", "ul", "ol", "table",
                                           "section", "article", "blockquote", "pre"};
            std::string tag;
            for (size_t j = i + 1; j < e && (std::isalnum((unsigned char)lo[j])); ++j)
                tag += lo[j];
            if (!tag.empty() || (i + 1 < e && lo[i + 1] == '/')) {
                std::string closing_tag;
                if (i + 1 < e && lo[i + 1] == '/')
                    for (size_t j = i + 2; j < e && std::isalnum((unsigned char)lo[j]); ++j)
                        closing_tag += lo[j];
                const std::string& t = tag.empty() ? closing_tag : tag;
                for (const char* b : blocks)
                    if (t == b) { text += '\n'; break; }
                if (t == "li") text += "\n- ";
                // Titres : préfixe markdown pour garder la structure (utile au skim CCR).
                if (!tag.empty() && tag.size() == 2 && tag[0] == 'h' && tag[1] >= '1' && tag[1] <= '3')
                    text += std::string((size_t)(tag[1] - '0') + 1, '#') + " ";
            }
            i = e + 1;
        } else {
            text += html[i++];
        }
    }
    decode_entities(text);

    // Compactage : espaces multiples, lignes vides en rafale, lignes de bruit.
    std::string out;
    out.reserve(text.size());
    int blank_run = 0;
    size_t line_start = 0;
    auto flush_line = [&](size_t b, size_t e) {
        // trim
        while (b < e && std::isspace((unsigned char)text[b])) ++b;
        while (e > b && std::isspace((unsigned char)text[e - 1])) --e;
        if (b >= e) { ++blank_run; return; }
        std::string line;
        bool sp = false;
        for (size_t i = b; i < e; ++i) {
            char c = text[i];
            if (c == ' ' || c == '\t' || c == '\r') { sp = true; continue; }
            if (sp && !line.empty()) line += ' ';
            sp = false;
            line += c;
        }
        if (blank_run > 0 && !out.empty()) out += '\n';
        blank_run = 0;
        out += line;
        out += '\n';
    };
    for (size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == '\n') {
            flush_line(line_start, i);
            line_start = i + 1;
        }
    }
    page.text = std::move(out);
    return page;
}

} // namespace laplace::research
