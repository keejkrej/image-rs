export type MenuItem =
  | {
      type: "item";
      id: string;
      label: string;
      shortcut?: string;
      enabled?: boolean;
    }
  | {
      type: "submenu";
      id: string;
      label: string;
      enabled?: boolean;
      items: MenuItem[];
    }
  | {
      type: "separator";
    };

export type TopLevelMenu = {
  id: string;
  label: string;
  items: MenuItem[];
};

export type MenuCommandId = string;

export type CommandDispatchEvent = {
  commandId: MenuCommandId;
  label: string;
  implemented: boolean;
  source: "menu" | "shortcut";
};
