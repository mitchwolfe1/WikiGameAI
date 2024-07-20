def remove_visited_nav_links(nav_links, visited):
    return {k: v for k, v in nav_links.items() if v not in visited}
